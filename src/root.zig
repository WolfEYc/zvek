const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

var allocator: Allocator = undefined;
// pub export fn add(a: i32, b: i32) i32 {
//     return a + b;
// }

// test "basic add functionality" {
//     try testing.expect(add(3, 7) == 10);
// }

fn print_vek(comptime T: type, v: Stream(T)) void {
    for (v.ptr) |x| {
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
    return struct {
        ptr: [*]Vek(T),
        start: usize = 0,
        stop: usize = 0,
    };
}
pub const Range = extern struct {
    start: usize = 0,
    stop: usize = 0,
};
pub fn Streams(comptime T: type) type {
    return extern struct {
        streams_ptr: [*]Vek(T),
        ranges_ptr: [*]Range,
        stream_len: usize,
        num_streams: usize,
    };
}

pub export fn init(buf_ptr: [*]u8, len: usize) void {
    const buf = buf_ptr[0..len];
    var fba = std.heap.FixedBufferAllocator.init(buf);
    allocator = fba.threadSafeAllocator();
}
pub fn Make_Result(comptime T: type) type {
    return extern union {
        ok: Streams(T),
        err: u16,
    };
}
pub fn make(comptime T: type, stream_len: usize, num_streams: usize) Make_Result(T) {
    const streams = allocator.alloc(Vek(T), stream_len * num_streams) catch |err| return Make_Result(T){ .err = @intFromError(err) };
    const ranges = allocator.alloc(Range, num_streams) catch |err| return Make_Result(T){ .err = @intFromError(err) };
    return Make_Result(T){
        .ok = Streams(T){
            .streams_ptr = streams.ptr,
            .ranges_ptr = ranges.ptr,
            .stream_len = stream_len,
            .num_streams = num_streams,
        },
    };
}
pub export fn make_f64(stream_len: usize, num_streams: usize) Make_Result(f64) {
    return make(f64, stream_len, num_streams);
}
pub export fn make_f32(stream_len: usize, num_streams: usize) Make_Result(f32) {
    return make(f32, stream_len, num_streams);
}
pub export fn make_i64(stream_len: usize, num_streams: usize) Make_Result(i64) {
    return make(i64, stream_len, num_streams);
}

pub fn get_stream(comptime T: type, streams: Streams(T), i: usize) Stream(T) {
    assert(i < streams.num_streams);
    const ptr_start = i * streams.stream_len;
    const ptr = streams.streams_ptr + ptr_start;
    const range = streams.ranges_ptr[i];
    return Stream(T){
        .ptr = ptr,
        .start = range.start,
        .stop = range.stop,
    };
}

pub const Operand_Variant = enum {
    Vector,
    Number,
};
pub fn Operand(comptime Num_T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => Stream(Num_T),
        .Number => Num_T,
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

pub fn apply(
    comptime Num_T: type,
    comptime op: Simd_Op,
    comptime mode: Apply_Mode,
    c: *Stream(Num_T),
    comptime a_t: Operand_Variant,
    a: Operand(Num_T, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(Num_T, b_t),
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
    const ninf: Vek(Num_T) = @splat(-std.math.inf(Num_T));

    const vek_start = c.start / lanes(Num_T);
    const vek_end = (c.stop - 1) / lanes(Num_T) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: Vek(Num_T) = switch (a_t) {
            .Vector => a.ptr[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(Num_T) = switch (b_t) {
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
            const a_nans = a_vek == ninf;
            c.ptr[i] = @select(Num_T, a_nans, b_vek, c.ptr[i]);
            const b_nans = b_vek == ninf;
            c.ptr[i] = @select(Num_T, b_nans, a_vek, c.ptr[i]);
        }
    }
}

pub export fn add(s: Streams(f64), c_idx: usize, a_idx: usize, b_idx: usize) void {
    const a = get_stream(f64, s, a_idx);
    const b = get_stream(f64, s, b_idx);
    var c = get_stream(f64, s, c_idx);
    apply(f64, .Add, .Strict, &c, .Vector, a, .Vector, b);
}
// pub export fn sub(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor(c: veks(f64), a: veks(f64), b: veks(f64)) void {
//     apply(f64, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

// pub export fn add_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Add, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn sub_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor_f32(c: veks(f32), a: veks(f32), b: veks(f32)) void {
//     apply(f32, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

// pub export fn add_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Add, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn sub_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn lshift_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .LShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn rshift_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .RShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn slshift_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .SLShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn and_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .And, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn or_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Or, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor_i64(c: veks(i64), a: veks(i64), b: veks(i64)) void {
//     apply(i64, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

// pub export fn add_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Add, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn sub_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn lshift_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .LShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn rshift_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .RShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn slshift_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .SLShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn and_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .And, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn or_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Or, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor_i32(c: veks(i32), a: veks(i32), b: veks(i32)) void {
//     apply(i32, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

// pub export fn add_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Add, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn sub_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn lshift_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .LShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn rshift_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .RShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn slshift_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .SLShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn and_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .And, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn or_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Or, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor_u64(c: veks(u64), a: veks(u64), b: veks(u64)) void {
//     apply(u64, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

// pub export fn add_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Add, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn sub_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Sub, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mul_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Mul, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn div_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Div, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn pow_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Pow, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn log_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Log, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn mod_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Mod, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn min_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Min, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn max_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Max, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn lshift_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .LShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn rshift_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .RShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn slshift_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .SLShift, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn and_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .And, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn or_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Or, .Strict, c, .Vector, a, .Vector, b);
// }
// pub export fn xor_u32(c: veks(u32), a: veks(u32), b: veks(u32)) void {
//     apply(u32, .Xor, .Strict, c, .Vector, a, .Vector, b);
// }

pub const BoolOp = enum {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
};

fn apply_bool(
    comptime num_t: type,
    comptime Op: BoolOp,
    comptime mode: Apply_Mode,
    c: *Stream(bool),
    comptime a_t: Operand_Variant,
    a: Operand(num_t, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(num_t, b_t),
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
    const ninf: Vek(num_t) = @splat(-std.math.inf(num_t));

    const vek_start = c.start / lanes(num_t);
    const vek_end = (c.stop - 1) / lanes(num_t) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: Vek(num_t) = switch (a_t) {
            .Vector => a.ptr[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(num_t) = switch (b_t) {
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
            const a_nans = a_vek == ninf;
            c.ptr[i] = @select(num_t, a_nans, b_vek, c.ptr[i]);
            const b_nans = b_vek == ninf;
            c.ptr[i] = @select(num_t, b_nans, a_vek, c.ptr[i]);
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
    const ninf = -std.math.inf(num_t);
    var a_arr = [_]Vek(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
    const a: Stream(num_t) = .{
        .ptr = &a_arr,
        .stop = 11,
    };
    var b_arr = [_]Vek(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
    const b: Stream(num_t) = .{
        .ptr = &b_arr,
        .stop = 11,
    };
    var c_expect_arr = [_]Vek(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, ninf, ninf, ninf, ninf, ninf } };
    const c_expect: Stream(num_t) = .{
        .ptr = &c_expect_arr,
        .stop = 11,
    };
    var c_actual_arr: [2]Vek(num_t) = undefined;
    @memset(&c_actual_arr, @splat(ninf));
    var c_actual: Stream(num_t) = .{
        .ptr = &c_actual_arr,
    };
    apply(num_t, .Add, .Strict, &c_actual, .Vector, a, .Vector, b);
    // print_vek(num_t, c_expect);
    // print_vek(num_t, c_actual);
    try testing.expectEqualSlices(Vek(num_t), c_expect.ptr, c_actual.ptr);
    try testing.expectEqual(c_expect.start, c_actual.start);
    try testing.expectEqual(c_expect.stop, c_actual.stop);
}

//TODO rewrite test w new framework
test "grow add" {
    const num_t = f64;
    const ninf = -std.math.inf(num_t);
    var a_arr = [_]Vek(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
    const a: Stream(num_t) = .{
        .ptr = &a_arr,
        .stop = 11,
    };
    var b_arr = [_]Vek(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, 30, ninf, ninf, ninf, ninf } };
    const b: Stream(num_t) = .{
        .ptr = &b_arr,
        .stop = 11,
    };
    var c_expect_arr = [_]Vek(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, 30, ninf, ninf, ninf, ninf } };
    const c_expect: Stream(num_t) = .{
        .ptr = &c_expect_arr,
        .stop = 11,
    };
    var c_actual_arr: [2]Vek(num_t) = undefined;
    @memset(&c_actual_arr, @splat(ninf));
    var c_actual: Stream(num_t) = .{
        .ptr = &c_actual_arr,
    };
    apply(num_t, .Add, .Grow, &c_actual, .Vector, a, .Vector, b);
    print_vek(num_t, c_expect);
    print_vek(num_t, c_actual);
    try testing.expectEqualSlices(Vek(num_t), c_expect.ptr, c_actual.ptr);
    try testing.expectEqual(c_expect.start, c_actual.start);
    try testing.expectEqual(c_expect.stop, c_actual.stop);
}
