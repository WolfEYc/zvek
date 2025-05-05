const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

// pub export fn add(a: i32, b: i32) i32 {
//     return a + b;
// }

// test "basic add functionality" {
//     try testing.expect(add(3, 7) == 10);
// }

fn print_vek(comptime T: type, v: Stream(T)) void {
    const iter = stream_iter(T, v);
    for (iter.start..iter.stop) |i| {
        std.debug.print("{d:.2} ", .{v.ptr[i]});
    }
    std.debug.print("\n", .{});
}

fn ceildiv(x: usize, y: usize) usize {
    return x / y + @intFromBool(x % y != 0);
}

fn print_float_slice(comptime T: type, s: []T) void {
    for (s) |x| {
        std.debug.print("{d:.2} ", .{x});
    }
    std.debug.print("\n", .{});
}

pub const TARGET_SIMD_SIZE: comptime_int = 64; // AVX512
pub fn lanes(comptime T: type) comptime_int {
    return TARGET_SIMD_SIZE / @sizeOf(T);
}

pub fn Vek(comptime T: type) type {
    return @Vector(lanes(T), T);
}
pub fn Stream(comptime T: type) type {
    return extern struct {
        ptr: [*]Vek(T),
        start: usize = 0,
        stop: usize = 0,
    };
}
pub const Range = extern struct {
    start: usize = 0,
    stop: usize = 0,
};
pub fn stream_iter(comptime T: type, stream: Stream(T)) Range {
    return Range{
        .start = stream.start / lanes(T),
        .stop = (stream.stop - 1) / lanes(T) + 1,
    };
}
pub fn Streams(comptime T: type) type {
    return extern struct {
        streams_ptr: [*]Vek(T),
        ranges_ptr: [*]Range,
        veks_per_stream: usize,
        len: usize,
        cap: usize,
    };
}

pub const State = struct {
    allocator: Allocator,
};
pub fn Result(comptime T: type) type {
    return extern struct {
        ok: T,
        err: u16 = 0,
    };
}

pub fn errify(comptime T: type, err: anyerror) Result(T) {
    return Result(T){ .ok = undefined, .err = @intFromError(err) };
}
pub fn resultify(comptime T: type, res: anyerror!T) Result(T) {
    if (res) |val| {
        return Result(T){ .ok = val };
    } else |err| {
        return errify(T, err);
    }
}

pub export fn init(buf_ptr: [*]u8, len: usize) Result(*State) {
    const buf = buf_ptr[0..len];
    var fba = std.heap.FixedBufferAllocator.init(buf);
    const allocator = fba.threadSafeAllocator();
    const state = allocator.create(State) catch |err| return errify(*State, err);
    state.allocator = allocator;
    return Result(*State){ .ok = state };
}

pub fn make(comptime T: type, allocator: Allocator, stream_size: usize, num_streams: usize) Allocator.Error!Streams(T) {
    const veks_per_stream = ceildiv(stream_size, lanes(T));
    const veks_cap = veks_per_stream * num_streams;
    const streams = try allocator.alloc(Vek(T), veks_cap);
    const ranges = try allocator.alloc(Range, num_streams);
    return Streams(T){
        .streams_ptr = streams.ptr,
        .ranges_ptr = ranges.ptr,
        .veks_per_stream = veks_per_stream,
        .len = 0,
        .cap = num_streams,
    };
}
pub export fn make_f64(s: *State, stream_len: usize, num_streams: usize) Result(Streams(f64)) {
    return resultify(Streams(f64), make(f64, s.allocator, stream_len, num_streams));
}
pub export fn make_f32(s: *State, stream_len: usize, num_streams: usize) Result(Streams(f32)) {
    return resultify(Streams(f32), make(f32, s.allocator, stream_len, num_streams));
}
pub export fn make_i64(s: *State, stream_len: usize, num_streams: usize) Result(Streams(i64)) {
    return resultify(Streams(i64), make(i64, s.allocator, stream_len, num_streams));
}

pub fn slice_to_stream(comptime T: type, streams: *Streams(T), slice: []T, start_idx: usize) Stream(T) {
    var stream = new_stream(T, streams);
    set_stream(T, &stream, slice, start_idx);
    return stream;
}

pub fn new_stream(comptime T: type, streams: *Streams(T)) Stream(T) {
    const stream = get_stream(T, streams, streams.len);
    streams.len += 1;
    return stream;
}
pub fn get_stream(comptime T: type, streams: *Streams(T), i: usize) Stream(T) {
    assert(i < streams.cap);
    const ptr_start = i * streams.veks_per_stream;
    const ptr = streams.streams_ptr + ptr_start;
    const range = streams.ranges_ptr[i];
    return Stream(T){
        .ptr = ptr,
        .start = range.start,
        .stop = range.stop,
    };
}
pub fn set_stream(comptime T: type, stream: *Stream(T), slice: []T, start_idx: usize) void {
    stream.start = start_idx;
    stream.stop = start_idx + slice.len;
    const stream_ptr: [*]T = @ptrCast(stream.ptr);
    @memcpy(stream_ptr + start_idx, slice);
}

pub const Operand_Variant = enum {
    Vector,
    Number,
};
pub fn Operand(comptime T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => Stream(T),
        .Number => T,
    };
}

pub const Apply_Mode = enum {
    Strict,
    Grow,
};
pub const Simd_Op = enum {
    Add,
    Sub,
    Div,
    Mul,
    Mod,
    Min,
    Max,
    Log,
    Pow,
    LShift,
    RShift,
    SLShift,
    And,
    Or,
    Xor,
};

fn idx_vek(comptime T: type) @Vector(lanes(T), usize) {
    var arr: [lanes(T)]usize = undefined;
    for (0..lanes(T)) |i| {
        arr[i] = i;
    }
    const vek: @Vector(lanes(T), usize) = arr;
    return vek;
}

fn nans_vek(comptime T: type, s: Stream(T), vek_i: usize) Vek(bool) {
    const index_vec = comptime idx_vek(T);
    const i = vek_i * lanes(T);
    const stop = @max(@min(s.stop - i, lanes(T)), 0);
    const start_idx = @max(@min(i - s.start, lanes(T)), 0);
    return (index_vec >= stop) | (index_vec < start_idx);
}

fn to_slice(comptime T: type, s: Stream(T)) []T {
    var slice: [*]T = @ptrCast(s.ptr);
    return slice[s.start..s.stop];
}

pub fn apply(
    comptime T: type,
    comptime op: Simd_Op,
    comptime mode: Apply_Mode,
    c: *Stream(T),
    comptime a_t: Operand_Variant,
    a: Operand(T, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(T, b_t),
) void {
    if (a_t == .Vector and b_t == .Vector) {
        assert(a.stop > a.start);
        assert(b.stop > b.start);
        c.start = switch (mode) {
            .Strict => @max(a.start, b.start),
            .Grow => @min(a.start, b.start),
        };
        c.stop = switch (mode) {
            .Strict => @min(a.stop, b.stop),
            .Grow => @max(a.stop, b.stop),
        };
    } else if (b_t == .Vector) {
        assert(b.stop > b.start);
        c.start = b.start;
        c.stop = b.stop;
    } else if (a_t == .Vector) {
        assert(a.stop > a.start);
        c.start = a.start;
        c.stop = a.stop;
    } else {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }

    const vek_start = c.start / lanes(T);
    const vek_end = (c.stop - 1) / lanes(T) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: Vek(T) = switch (a_t) {
            .Vector => a.ptr[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Vector => b.ptr[i],
            .Number => @splat(b),
        };
        c.ptr[i] = switch (op) {
            .Add => a_vek + b_vek,
            .Sub => a_vek - b_vek,
            .Div => a_vek / b_vek,
            .Mul => a_vek * b_vek,
            .Mod => a_vek % b_vek,
            .Min => @min(a_vek, b_vek),
            .Max => @max(a_vek, b_vek),
            .Log => @log2(b_vek) / @log2(a_vek),
            .Pow => @exp2(b_vek * @log2(a_vek)),
            .LShift => a_vek << b_vek,
            .RShift => a_vek >> b_vek,
            .SLShift => a_vek <<| b_vek,
            .And => a_vek & b_vek,
            .Or => a_vek | b_vek,
            .Xor => a_vek ^ b_vek,
        };
        if (mode == .Grow) {
            const a_nans = nans_vek(T, a, i);
            c.ptr[i] = @select(T, a_nans, b_vek, c.ptr[i]);
            const b_nans = nans_vek(T, b, i);
            c.ptr[i] = @select(T, b_nans, a_vek, c.ptr[i]);
        }
    }
}

pub const BoolOp = enum {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
};

fn apply_bool(
    comptime T: type,
    comptime Op: BoolOp,
    comptime mode: Apply_Mode,
    c: *Stream(bool),
    comptime a_t: Operand_Variant,
    a: Operand(T, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(T, b_t),
) void {
    if (a_t == .Vector and b_t == .Vector) {
        assert(a.stop > a.start);
        assert(b.stop > b.start);
        c.start = switch (mode) {
            .Strict => @max(a.start, b.start),
            .Grow => @min(a.start, b.start),
        };
        c.stop = switch (mode) {
            .Strict => @min(a.stop, b.stop),
            .Grow => @max(a.stop, b.stop),
        };
    } else if (b_t == .Vector) {
        assert(b.stop > b.start);
        c.start = b.start;
        c.stop = b.stop;
    } else if (a_t == .Vector) {
        assert(a.stop > a.start);
        c.start = a.start;
        c.stop = a.stop;
    } else {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }

    const vek_start = c.start / lanes(T);
    const vek_end = (c.stop - 1) / lanes(T) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: Vek(T) = switch (a_t) {
            .Vector => a.ptr[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Vector => b.ptr[i],
            .Number => @splat(b),
        };
        c.ptr[i] = switch (Op) {
            .Gt => a_vek > b_vek,
            .Gte => a_vek >= b_vek,
            .Lt => a_vek < b_vek,
            .Lte => a_vek <= b_vek,
            .Eq => a_vek == b_vek,
            .Neq => a_vek != b_vek,
        };
        if (mode == .Grow) {
            const a_nans = nans_vek(T, a, i);
            c.ptr[i] = @select(T, a_nans, b_vek, c.ptr[i]);
            const b_nans = nans_vek(T, b, i);
            c.ptr[i] = @select(T, b_nans, a_vek, c.ptr[i]);
        }
    }
}

pub const SimdOp1 = enum {
    Ceil,
    Floor,
    Round,
    Sqrt,
    Not,
    Neg,
};

pub fn apply_single(
    comptime num_t: type,
    comptime Op: SimdOp1,
    c: *Stream(num_t),
) void {
    const vek_start = c.start / lanes(num_t);
    const vek_end = (c.stop - 1) / lanes(num_t) + 1;
    for (vek_start..vek_end) |i| {
        c.ptr[i] = switch (Op) {
            .Ceil => @ceil(c.ptr[i]),
            .Floor => @floor(c.ptr[i]),
            .Round => @round(c.ptr[i]),
            .Sqrt => @sqrt(c.ptr[i]),
            .Not => ~c.ptr[i],
            .Neg => -c.ptr[i],
        };
    }
}

// TODO
pub const SimdOp3 = enum {
    Fma,
};

//TODO rewrite test w new framework
test "basic add" {
    const num_t = f64;

    var beig_buffer: [1024]u8 = undefined;
    const state_res = init(&beig_buffer, 1024);
    if (state_res.err != 0) {
        unreachable;
    }
    const state = state_res.ok;
    var streams = try make(num_t, state.allocator, 11, 4);

    var a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const a_stream = slice_to_stream(num_t, &streams, &a_arr, 0);

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const b_stream = slice_to_stream(num_t, &streams, &b_arr, 0);

    var c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    const c_expect_stream = slice_to_stream(num_t, &streams, &c_expect_arr, 0);

    var c_actual_stream = new_stream(num_t, &streams);
    apply(num_t, .Add, .Strict, &c_actual_stream, .Vector, a_stream, .Vector, b_stream);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    try testing.expectEqualSlices(num_t, to_slice(num_t, c_expect_stream), to_slice(num_t, c_actual_stream));
}

// //TODO rewrite test w new framework
// test "grow add" {
//     const num_t = f64;
//     const ninf = -std.math.inf(num_t);
//     var a_arr = [_]Vek(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
//     const a: Stream(num_t) = .{
//         .ptr = &a_arr,
//         .stop = 11,
//     };
//     var b_arr = [_]Vek(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, 30, ninf, ninf, ninf, ninf } };
//     const b: Stream(num_t) = .{
//         .ptr = &b_arr,
//         .stop = 11,
//     };
//     var c_expect_arr = [_]Vek(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, 30, ninf, ninf, ninf, ninf } };
//     const c_expect: Stream(num_t) = .{
//         .ptr = &c_expect_arr,
//         .stop = 11,
//     };
//     var c_actual_arr: [2]Vek(num_t) = undefined;
//     @memset(&c_actual_arr, @splat(ninf));
//     var c_actual: Stream(num_t) = .{
//         .ptr = &c_actual_arr,
//     };
//     apply(num_t, .Add, .Grow, &c_actual, .Vector, a, .Vector, b);
//     print_vek(num_t, c_expect);
//     print_vek(num_t, c_actual);
//     try testing.expectEqualSlices(Vek(num_t), c_expect.ptr, c_actual.ptr);
//     try testing.expectEqual(c_expect.start, c_actual.start);
//     try testing.expectEqual(c_expect.stop, c_actual.stop);
// }

// #region EXPORTS!!!
pub export fn add(c: *Stream(f64), a: Stream(f64), b: Stream(f64)) void {
    apply(f64, .Add, .Strict, c, .Vector, a, .Vector, b);
}
