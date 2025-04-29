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

fn print_vek(comptime T: type, v: veks(T)) void {
    for (v.veks) |x| {
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

fn lanes(comptime T: type) usize {
    return TARGET_SIMD_SIZE / @sizeOf(T);
}

fn vek_t(comptime T: type) type {
    return @Vector(lanes(T), T);
}
fn veks(comptime T: type) type {
    return struct {
        veks: []vek_t(T),
        start: usize = 0,
        stop: usize = 0,
    };
}

/// apply but will effectively be a memcpy for non overlaping indexes
pub fn apply_grow(comptime T: type, comptime Op: SimdOp, c: *veks(T), a: veks(T), b: veks(T)) void {
    assert(a.stop > a.start);
    assert(b.stop > b.start);
    assert(a.veks.len == b.veks.len and b.veks.len == c.veks.len);
    c.start = @min(a.start, b.start);
    c.stop = @max(a.stop, b.stop);

    //TODO https://ziglang.org/documentation/0.14.0/#select <-- use this guy to snag da crooked bytes

    const vek_start = c.start / lanes(T);
    const vek_end = (c.stop - 1) / lanes(T) + 1;

    for (vek_start..vek_end) |i| {
        c.veks[i] = switch (Op) {
            .Add => a.veks[i] + b.veks[i],
            .Sub => a.veks[i] - b.veks[i],
            .Div => a.veks[i] / b.veks[i],
            .Mul => a.veks[i] * b.veks[i],
            .Mod => a.veks[i] % b.veks[i],
            .Min => @min(a.veks[i], b.veks[i]),
            .Max => @max(a.veks[i], b.veks[i]),
            .Log => @log2(b.veks[i]) / @log2(a.veks[i]),
            .Pow => @exp2(b.veks[i] * @log2(a.veks[i])),
            .LShift => a.veks[i] << b.veks[i],
            .RShift => a.veks[i] >> b.veks[i],
            .SLShift => a.veks[i] <<| b.veks[i],
            .And => a.veks[i] & b.veks[i],
            .Or => a.veks[i] | b.veks[i],
            .Xor => a.veks[i] ^ b.veks[i],
        };
    }
}

pub fn apply_strict(comptime T: type, comptime Op: SimdOp, c: *veks(T), a: veks(T), b: veks(T)) void {
    assert(a.stop > a.start);
    assert(b.stop > b.start);
    assert(a.veks.len == b.veks.len and b.veks.len == c.veks.len);
    c.start = @max(a.start, b.start);
    c.stop = @min(a.stop, b.stop);

    const vek_start = c.start / lanes(T);
    const vek_end = (c.stop - 1) / lanes(T) + 1;

    for (vek_start..vek_end) |i| {
        c.veks[i] = switch (Op) {
            .Add => a.veks[i] + b.veks[i],
            .Sub => a.veks[i] - b.veks[i],
            .Div => a.veks[i] / b.veks[i],
            .Mul => a.veks[i] * b.veks[i],
            .Mod => a.veks[i] % b.veks[i],
            .Min => @min(a.veks[i], b.veks[i]),
            .Max => @max(a.veks[i], b.veks[i]),
            .Log => @log2(b.veks[i]) / @log2(a.veks[i]),
            .Pow => @exp2(b.veks[i] * @log2(a.veks[i])),
            .LShift => a.veks[i] << b.veks[i],
            .RShift => a.veks[i] >> b.veks[i],
            .SLShift => a.veks[i] <<| b.veks[i],
            .And => a.veks[i] & b.veks[i],
            .Or => a.veks[i] | b.veks[i],
            .Xor => a.veks[i] ^ b.veks[i],
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

pub const SimdOp3 = enum {
    Fma,
};

test "basic add" {
    const num_t = f64;
    var a_arr = [_]vek_t(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, 0, 0, 0, 0, 0 } };
    const a: veks(num_t) = .{
        .veks = &a_arr,
        .stop = 11,
    };
    var b_arr = [_]vek_t(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, 0, 0, 0, 0, 0 } };
    const b: veks(num_t) = .{
        .veks = &b_arr,
        .stop = 11,
    };
    var c_expect_arr = [_]vek_t(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, 0, 0, 0, 0, 0 } };
    const c_expect: veks(num_t) = .{
        .veks = &c_expect_arr,
        .stop = 11,
    };
    var c_actual_arr = std.mem.zeroes([2]vek_t(num_t));
    var c_actual: veks(num_t) = .{
        .veks = &c_actual_arr,
    };
    apply(num_t, .Add, &c_actual, a, b);
    print_vek(num_t, c_expect);
    print_vek(num_t, c_actual);
    try testing.expectEqualSlices(vek_t(num_t), c_expect.veks, c_actual.veks);
    try testing.expectEqual(c_expect.start, c_actual.start);
    try testing.expectEqual(c_expect.stop, c_actual.stop);
}

test "misaligned add" {
    const num_t = f64;
    var a_arr = [_]vek_t(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, 0, 0, 0, 0, 0 } };
    const a: veks(num_t) = .{
        .veks = &a_arr,
        .stop = 11,
    };
    var b_arr = [_]vek_t(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, 0, 0, 0, 0, 0 } };
    const b: veks(num_t) = .{
        .veks = &b_arr,
        .stop = 11,
    };
    var c_expect_arr = [_]vek_t(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, 0, 0, 0, 0, 0 } };
    const c_expect: veks(num_t) = .{
        .veks = &c_expect_arr,
        .stop = 11,
    };
    var c_actual_arr = std.mem.zeroes([2]vek_t(num_t));
    var c_actual: veks(num_t) = .{
        .veks = &c_actual_arr,
    };
    apply(num_t, .Add, &c_actual, a, b);
    print_vek(num_t, c_expect);
    print_vek(num_t, c_actual);
    try testing.expectEqualSlices(vek_t(num_t), c_expect.veks, c_actual.veks);
    try testing.expectEqual(c_expect.start, c_actual.start);
    try testing.expectEqual(c_expect.stop, c_actual.stop);
}
