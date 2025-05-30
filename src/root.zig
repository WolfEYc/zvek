const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

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
    // LShift,
    // RShift,
    // SLShift,
    And,
    Or,
    Xor,
};

const simd_float_ops = [_]Simd_Op{
    .Log,
    .Pow,
};

const number_types = [_]type{
    u8,
    u16,
    u32,
    u64,
    i8,
    i16,
    i32,
    i64,
    f32,
    f64,
};

const float_types = [_]type{
    f32, f64,
};

pub const Operand_Variant = enum {
    Vector,
    Number,
};

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

pub const TARGET_SIMD_BITS: comptime_int = 512; // AVX512
pub fn lanes(comptime T: type) comptime_int {
    return TARGET_SIMD_BITS / 8 / @sizeOf(T);
}

pub fn Vek(comptime T: type) type {
    return @Vector(lanes(T), T);
}
pub fn Stream(comptime T: type) type {
    return extern struct {
        ptr: [*]Vek(T),
        len: usize = 0,
    };
}
pub const StreamAllocator = extern struct { allocator: Allocator };
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

pub fn make(buf: []u8) StreamAllocator {
    var arena = std.heap.FixedBufferAllocator.init(buf);
    return StreamAllocator{ .allocator = arena.allocator() };
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

pub fn to_stream(comptime T: type, slice: []T) Stream(T) {
    var stream = new_stream(T, slice.len);
    set_stream(T, &stream, slice, start_idx);
    return stream;
}

pub fn new_stream(comptime T: type, size: usize) Stream(T) {
    return Stream(T){
        .ptr = ptr,
        .start = range.start,
    };
}
pub fn set_stream(comptime T: type, stream: *Stream(T), slice: []T, start_idx: usize) void {
    stream.start = start_idx;
    stream.stop = start_idx + slice.len;
    const stream_ptr: [*]T = @ptrCast(stream.ptr);
    @memcpy(stream_ptr + start_idx, slice);
}

pub fn Operand(comptime T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => Stream(T),
        .Number => T,
    };
}

fn idx_vek(comptime T: type) @Vector(lanes(T), usize) {
    return comptime ret: {
        var arr: [lanes(T)]usize = undefined;
        for (0..lanes(T)) |i| {
            arr[i] = i;
        }
        break :ret arr;
    };
}

fn nans_vek(comptime T: type, start_idx: usize, stop: usize, vek_i: usize) @Vector(lanes(T), bool) {
    const i: usize = vek_i * lanes(T);
    const i_vek: @Vector(lanes(T), usize) = @splat(i);
    const index_vec = idx_vek(T) + i_vek;
    const start_idx_vek: @Vector(lanes(T), usize) = @splat(start_idx);
    const stop_vek: @Vector(lanes(T), usize) = @splat(stop);
    const lhs: @Vector(lanes(T), u1) = @bitCast(index_vec >= stop_vek);
    const rhs: @Vector(lanes(T), u1) = @bitCast(index_vec < start_idx_vek);
    const res: @Vector(lanes(T), bool) = @bitCast(lhs | rhs);
    return res;
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
            .Mod => @mod(a_vek, b_vek),
            .Min => @min(a_vek, b_vek),
            .Max => @max(a_vek, b_vek),
            .Log => @log2(b_vek) / @log2(a_vek),
            .Pow => @exp2(b_vek * @log2(a_vek)),
            // .LShift => a_vek << b_vek,
            // .RShift => a_vek >> b_vek,
            // .SLShift => a_vek <<| b_vek,
            .And, .Or, .Xor => blk: {
                const a_bits: @Vector(TARGET_SIMD_BITS, u1) = @bitCast(a_vek);
                const b_bits: @Vector(TARGET_SIMD_BITS, u1) = @bitCast(b_vek);
                const c_bits: @Vector(TARGET_SIMD_BITS, u1) = switch (op) {
                    .And => a_bits & b_bits,
                    .Or => a_bits | b_bits,
                    .Xor => a_bits ^ b_bits,
                    else => comptime unreachable,
                };
                const c_vek: Vek(T) = @bitCast(c_bits);
                break :blk c_vek;
            },
        };
        if (mode == .Grow) {
            if (a_t == .Vector) {
                const a_nans = nans_vek(T, a.start, a.stop, i);
                c.ptr[i] = @select(T, a_nans, b_vek, c.ptr[i]);
            }
            if (b_t == .Vector) {
                const b_nans = nans_vek(T, b.start, b.stop, i);
                c.ptr[i] = @select(T, b_nans, a_vek, c.ptr[i]);
            }
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

pub fn apply_bool(
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
            const a_nans = nans_vek(T, a.start, a.stop, i);
            c.ptr[i] = @select(T, a_nans, b_vek, c.ptr[i]);
            const b_nans = nans_vek(T, b.start, b.stop, i);
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
    const a_stream = to_stream(num_t, &streams, &a_arr, 0);

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const b_stream = to_stream(num_t, &streams, &b_arr, 0);

    var c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    const c_expect_stream = to_stream(num_t, &streams, &c_expect_arr, 0);

    var c_actual_stream = new_stream(num_t, &streams);
    apply(num_t, .Add, .Strict, &c_actual_stream, .Vector, a_stream, .Vector, b_stream);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    try testing.expectEqualSlices(num_t, to_slice(num_t, c_expect_stream), to_slice(num_t, c_actual_stream));
}

test "basic growing add" {
    const num_t = f64;

    var beig_buffer: [1024]u8 = undefined;
    const state_res = init(&beig_buffer, 1024);
    if (state_res.err != 0) {
        unreachable;
    }
    const state = state_res.ok;
    var streams = try make(num_t, state.allocator, 12, 4);

    var a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const a_stream = to_stream(num_t, &streams, &a_arr, 0);

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20, 21 };
    const b_stream = to_stream(num_t, &streams, &b_arr, 0);

    var c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40, 21 };
    const c_expect_stream = to_stream(num_t, &streams, &c_expect_arr, 0);

    var c_actual_stream = new_stream(num_t, &streams);
    apply(num_t, .Add, .Grow, &c_actual_stream, .Vector, a_stream, .Vector, b_stream);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    try testing.expectEqualSlices(num_t, to_slice(num_t, c_expect_stream), to_slice(num_t, c_actual_stream));
}

// #region EXPORTS!!!
// pub export fn add(c: *Stream(f64), a: Stream(f64), b: Stream(f64)) void {
//     apply(f64, .Add, .Strict, c, .Vector, a, .Vector, b);
// }

comptime {
    @setEvalBranchQuota(4096);
    for (@typeInfo(Apply_Mode).@"enum".fields) |mode| {
        const mode_enum: Apply_Mode = @enumFromInt(mode.value);
        for (number_types) |t| {
            var is_float_t = false;
            for (float_types) |float_ts| {
                if (t != float_ts) continue;
                is_float_t = true;
                break;
            }
            op_loop: for (@typeInfo(Simd_Op).@"enum".fields) |op| {
                const op_enum: Simd_Op = @enumFromInt(op.value);
                if (!is_float_t) {
                    for (simd_float_ops) |float_op| {
                        if (op_enum == float_op) {
                            continue :op_loop;
                        }
                    }
                }
                generate_apply_func(t, op_enum, mode_enum, .Vector, .Vector);
                generate_apply_func(t, op_enum, mode_enum, .Vector, .Number);
                generate_apply_func(t, op_enum, mode_enum, .Number, .Vector);
            }
        }
    }
}

fn generate_apply_func(
    comptime T: type,
    comptime op: Simd_Op,
    comptime mode: Apply_Mode,
    comptime a_t: Operand_Variant,
    comptime b_t: Operand_Variant,
) void {
    var name: []const u8 = @tagName(op) ++ "_" ++ @typeName(T);
    if (a_t == .Number) {
        name = "num_" ++ name;
    }
    if (b_t == .Number) {
        name = name ++ "_num";
    }
    if (mode != .Strict) {
        name = name ++ "_" ++ @tagName(mode);
    }
    @export(&struct {
        fn generated(c: *Stream(T), a: Operand(T, a_t), b: Operand(T, b_t)) callconv(.C) void {
            apply(T, op, mode, c, a_t, a, b_t, b);
        }
    }.generated, .{ .name = name });
}
