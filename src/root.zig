const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;

// pub export fn add(a: i32, b: i32) i32 {
//     return a + b;
// }

// test "basic add functionality" {
//     try testing.expect(add(3, 7) == 10);
// }

pub const TARGET_SIMD_SIZE: usize = 64; // AVX512

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

pub fn lanes(comptime T: type) comptime_int {
    return TARGET_SIMD_SIZE / @sizeOf(T);
}

pub fn vek_t(comptime T: type) type {
    return @Vector(lanes(T), T);
}
pub fn veks(comptime T: type) type {
    return struct {
        veks: []vek_t(T),
        start: usize = 0,
        stop: usize = 0,
    };
}

pub const Operand_T = enum {
    vector,
    number,
};
pub fn operand(comptime num_t: type, comptime Variant: Operand_T) type {
    return switch (Variant) {
        .vector => veks(num_t),
        .number => num_t,
    };
}

pub const ApplyMode = enum {
    Strict,
    Grow,
};

pub fn apply(
    comptime num_t: type,
    comptime Op: SimdOp,
    comptime mode: ApplyMode,
    c: *veks(num_t),
    comptime a_t: Operand_T,
    a: operand(num_t, a_t),
    comptime b_t: Operand_T,
    b: operand(num_t, b_t),
) void {
    if (a_t == .vector and b_t == .vector) {
        assert(a.stop > a.start);
        assert(b.stop > b.start);
        assert(a.veks.len == b.veks.len and b.veks.len == c.veks.len);
        c.start = switch (mode) {
            .Strict => @max(a.start, b.start),
            .Grow => @min(a.start, b.start),
        };
        c.stop = switch (mode) {
            .Strict => @min(a.stop, b.stop),
            .Grow => @max(a.stop, b.stop),
        };
    } else if (b_t == .vector) {
        assert(b.stop > b.start);
        assert(b.veks.len == c.veks.len);
        c.start = b.start;
        c.stop = b.stop;
    } else if (a_t == .vector) {
        assert(a.stop > a.start);
        assert(a.veks.len == c.veks.len);
        c.start = a.start;
        c.stop = a.stop;
    } else {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    const ninf: vek_t(num_t) = @splat(-std.math.inf(num_t));

    const vek_start = c.start / lanes(num_t);
    const vek_end = (c.stop - 1) / lanes(num_t) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: vek_t(num_t) = switch (a_t) {
            .vector => a.veks[i],
            .number => @splat(a),
        };
        const b_vek: vek_t(num_t) = switch (b_t) {
            .vector => b.veks[i],
            .number => @splat(b),
        };
        c.veks[i] = switch (Op) {
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
            c.veks[i] = @select(num_t, a_nans, b_vek, c.veks[i]);
            const b_nans = b_vek == ninf;
            c.veks[i] = @select(num_t, b_nans, a_vek, c.veks[i]);
        }
    }
}

pub const SimdOp1 = enum {
    Ceil,
    Floor,
    Round,
    Sqrt,
    Not,
};

pub fn apply_single(
    comptime num_t: type,
    comptime Op: SimdOp1,
    c: *veks(num_t),
) void {
    const vek_start = c.start / lanes(num_t);
    const vek_end = (c.stop - 1) / lanes(num_t) + 1;
    for (vek_start..vek_end) |i| {
        c.veks[i] = switch (Op) {
            .Ceil => @ceil(c.veks[i]),
            .Floor => @floor(c.veks[i]),
            .Round => @round(c.veks[i]),
            .Sqrt => @sqrt(c.veks[i]),
            .Not => ~c.veks[i],
        };
    }
}

// TODO
pub const SimdOp3 = enum {
    Fma,
};

test "basic add" {
    const num_t = f64;
    const ninf = -std.math.inf(num_t);
    var a_arr = [_]vek_t(num_t){ .{ 0, 1, 7, 69, 420, 666, 6969, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
    const a: veks(num_t) = .{
        .veks = &a_arr,
        .stop = 11,
    };
    var b_arr = [_]vek_t(num_t){ .{ 0, 1, 3, 96, 580, 444, 9696, 50 }, .{ 100, 10, 20, ninf, ninf, ninf, ninf, ninf } };
    const b: veks(num_t) = .{
        .veks = &b_arr,
        .stop = 11,
    };
    var c_expect_arr = [_]vek_t(num_t){ .{ 0, 2, 10, 165, 1000, 1110, 16665, 100 }, .{ 200, 20, 40, ninf, ninf, ninf, ninf, ninf } };
    const c_expect: veks(num_t) = .{
        .veks = &c_expect_arr,
        .stop = 11,
    };
    var c_actual_arr: [2]vek_t(num_t) = undefined;
    @memset(&c_actual_arr, @splat(ninf));
    var c_actual: veks(num_t) = .{
        .veks = &c_actual_arr,
    };
    apply(num_t, .Add, .Strict, &c_actual, .vector, a, .vector, b);
    // print_vek(num_t, c_expect);
    // print_vek(num_t, c_actual);
    try testing.expectEqualSlices(vek_t(num_t), c_expect.veks, c_actual.veks);
    try testing.expectEqual(c_expect.start, c_actual.start);
    try testing.expectEqual(c_expect.stop, c_actual.stop);
}
