const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;

// pub export fn add(a: i32, b: i32) i32 {
//     return a + b;
// }

// test "basic add functionality" {
//     try testing.expect(add(3, 7) == 10);
// }

const TARGET_SIMD_SIZE: usize = 64; // AVX512

fn print_f64s(s: []const f64) void {
    for (s) |x| {
        std.debug.print("{d:.2} ", .{x});
    }
    std.debug.print("\n", .{});
}
fn print_f32s(s: []const f32) void {
    for (s) |x| {
        std.debug.print("{d:.2} ", .{x});
    }
    std.debug.print("\n", .{});
}

pub const SimdOp = enum {
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

//TODO add 64 bit aligned metric buffer allocation funcs
// make all simd funcs operate over aligned ptrCasts

pub fn apply(comptime T: type, comptime Op: SimdOp, c: []T, a: []const T, b: []const T) void {
    assert(a.len == b.len and b.len == c.len);
    const lanes = TARGET_SIMD_SIZE / @sizeOf(T);
    const vec_t = @Vector(lanes, T);
    const leftovers = a.len % lanes;
    const stop = a.len - leftovers;
    var i: usize = 0;

    while (i < stop) : (i += lanes) {
        const a_vec: vec_t = @ptrCast(a[i..][0..lanes].ptr); // does not work rn
        const b_vec: vec_t = @ptrCast(b[i..][0..lanes].ptr);
        c[i..][0..lanes].* = switch (Op) {
            .Add => a_vec + b_vec,
            .Sub => a_vec - b_vec,
            .Div => a_vec / b_vec,
            .Mul => a_vec * b_vec,
            .Mod => a_vec % b_vec,
            .Min => @min(a_vec, b_vec),
            .Max => @max(a_vec, b_vec),
            .Log => @log2(b_vec) / @log2(a_vec),
            .Pow => @exp2(b_vec * @log2(a_vec)),
            .LShift => a_vec << b_vec,
            .RShift => a_vec >> b_vec,
            .SLShift => a_vec <<| b_vec,
            .And => a_vec & b_vec,
            .Or => a_vec | b_vec,
            .Xor => a_vec ^ b_vec,
        };
    }
    // leftovers
    while (i < a.len) : (i += 1) {
        c[i] = switch (Op) {
            .Add => a[i] + b[i],
            .Sub => a[i] - b[i],
            .Div => a[i] / b[i],
            .Mul => a[i] * b[i],
            .Mod => a[i] % b[i],
            .Min => @min(a[i], b[i]),
            .Max => @max(a[i], b[i]),
            .Log => @log2(b[i]) / @log2(a[i]),
            .Pow => @exp2(b[i] * @log2(a[i])),
            .LShift => a[i] << b[i],
            .RShift => a[i] >> b[i],
            .SLShift => a[i] <<| b[i],
            .And => a[i] & b[i],
            .Or => a[i] | b[i],
            .Xor => a[i] ^ b[i],
        };
    }
}

pub fn apply_num(comptime T: type, comptime Op: SimdOp, c: []T, a: []const T, b: T) void {
    assert(a.len == c.len);
    const lanes = TARGET_SIMD_SIZE / @sizeOf(T);
    const vec_t = @Vector(lanes, T);
    const leftovers = a.len % lanes;
    const stop = a.len - leftovers;
    var i: usize = 0;

    const b_vec: vec_t = @splat(b);
    while (i < stop) : (i += lanes) {
        const a_vec: vec_t = a[i..][0..lanes].*;
        c[i..][0..lanes].* = switch (Op) {
            .Add => a_vec + b_vec,
            .Sub => a_vec - b_vec,
            .Div => a_vec / b_vec,
            .Mul => a_vec * b_vec,
            .Mod => a_vec % b_vec,
            .Min => @min(a_vec, b_vec),
            .Max => @max(a_vec, b_vec),
            .Log => @log2(b_vec) / @log2(a_vec),
            .Pow => @exp2(b_vec * @log2(a_vec)),
            .LShift => a_vec << b_vec,
            .SLShift => a_vec <<| b_vec,
            .RShift => a_vec >> b_vec,
            .And => a_vec & b_vec,
            .Or => a_vec | b_vec,
            .Xor => a_vec ^ b_vec,
        };
    }
    // leftovers
    while (i < a.len) : (i += 1) {
        c[i] = switch (Op) {
            .Add => a[i] + b,
            .Sub => a[i] - b,
            .Div => a[i] / b,
            .Mul => a[i] * b,
            .Mod => a[i] % b,
            .Min => @min(a[i], b),
            .Max => @max(a[i], b),
            .Log => @log2(b) / @log2(a[i]),
            .Pow => @exp2(b * @log2(a[i])),
            .LShift => a[i] << b,
            .SLShift => a[i] <<| b,
            .RShift => a[i] >> b,
            .And => a[i] & b,
            .Or => a[i] | b,
            .Xor => a[i] ^ b,
        };
    }
}

fn num_apply(comptime T: type, comptime Op: SimdOp, c: []T, a: T, b: []const T) void {
    assert(b.len == c.len);
    const lanes = TARGET_SIMD_SIZE / @sizeOf(T);
    const vec_t = @Vector(lanes, T);
    const leftovers = b.len % lanes;
    const stop = b.len - leftovers;
    var i: usize = 0;

    const a_vec: vec_t = @splat(a);
    while (i < stop) : (i += lanes) {
        const b_vec: vec_t = b[i..][0..lanes].*;
        c[i..][0..lanes].* = switch (Op) {
            .Add => a_vec + b_vec,
            .Sub => a_vec - b_vec,
            .Div => a_vec / b_vec,
            .Mul => a_vec * b_vec,
            .Mod => a_vec % b_vec,
            .Min => @min(a_vec, b_vec),
            .Max => @max(a_vec, b_vec),
            .Log => @log2(b_vec) / @log2(a_vec),
            .Pow => @exp2(b_vec * @log2(a_vec)),
            .LShift => a_vec << b_vec,
            .SLShift => a_vec <<| b_vec,
            .RShift => a_vec >> b_vec,
            .And => a_vec & b_vec,
            .Or => a_vec | b_vec,
            .Xor => a_vec ^ b_vec,
        };
    }
    // leftovers
    while (i < a.len) : (i += 1) {
        c[i] = switch (Op) {
            .Add => a + b[i],
            .Sub => a - b[i],
            .Div => a / b[i],
            .Mul => a * b[i],
            .Mod => a % b[i],
            .Min => @min(a, b[i]),
            .Max => @max(a, b[i]),
            .Log => @log2(b[i]) / @log2(a),
            .Pow => @exp2(b[i] * @log2(a)),
            .LShift => a << b[i],
            .SLShift => a <<| b[i],
            .RShift => a >> b[i],
            .And => a & b[i],
            .Or => a | b[i],
            .Xor => a ^ b[i],
        };
    }
}

pub const SimdOp1 = enum {
    Ceil,
    Floor,
    Round,
    Sqrt,
    Not,
};

pub fn apply1(comptime T: type, comptime Op: SimdOp1, c: []T, a: []const T) void {
    assert(a.len == c.len);
    const lanes = TARGET_SIMD_SIZE / @sizeOf(T);
    const vec_t = @Vector(lanes, T);
    const leftovers = a.len % lanes;
    const stop = a.len - leftovers;
    var i: usize = 0;
    while (i < stop) : (i += lanes) {
        const a_vec: vec_t = a[i..][0..lanes].*;
        c[i..][0..lanes].* = switch (Op) {
            .Ceil => @ceil(a_vec),
            .Floor => @floor(a_vec),
            .Round => @round(a_vec),
            .Sqrt => @sqrt(a_vec),
            .Not => ~a_vec,
        };
    }
    // leftovers
    while (i < a.len) : (i += 1) {
        c[i] = switch (Op) {
            .Ceil => @ceil(a[i]),
            .Floor => @floor(a[i]),
            .Round => @round(a[i]),
            .Sqrt => @sqrt(a[i]),
            .Not => ~a[i],
        };
    }
}

pub const SimdOp3 = enum {
    Fma,
};
fn apply3(comptime T: type, comptime Op: SimdOp3, d: []T, a: []const T, b: []const T, c: []const T) void {
    assert(a.len == b.len and b.len == c.len);
    const lanes = TARGET_SIMD_SIZE / @sizeOf(T);
    const vec_t = @Vector(lanes, T);
    const leftovers = a.len % lanes;
    const stop = a.len - leftovers;
    var i: usize = 0;

    while (i < stop) : (i += lanes) {
        const a_vec: vec_t = a[i..][0..lanes].*;
        const b_vec: vec_t = b[i..][0..lanes].*;
        const c_vec: vec_t = c[i..][0..lanes].*;
        d[i..][0..lanes].* = switch (Op) {
            .Fma => @mulAdd(vec_t, a_vec, b_vec, c_vec),
        };
    }
    // leftovers
    while (i < a.len) : (i += 1) {
        d[i] = switch (Op) {
            .Add => a[i] + b[i],
        };
    }
}

test "add some shit!!! YUUUH" {
    const a: []const f64 = &.{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const b: []const f64 = &.{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const c_expect: []const f64 = &.{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    var c_mem: [11]f64 = undefined;
    var known_at_runtime_zero: usize = 0;
    _ = &known_at_runtime_zero;
    const c_actual = c_mem[known_at_runtime_zero..];
    apply(f64, .Add, c_actual, a, b);
    // print_f64s(c_expect);
    // print_f64s(c_actual);
    try testing.expectEqualSlices(f64, c_expect, c_actual);
}
