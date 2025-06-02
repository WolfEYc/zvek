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
    const stream_ptr: [*]T = @ptrCast(v.veks);
    for (0..v.len_scalars) |i| {
        std.debug.print("{d:.2} ", .{stream_ptr[i]});
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
pub fn lanes(comptime T: type) usize {
    return TARGET_SIMD_BITS / 8 / @sizeOf(T);
}

pub fn Vek(comptime T: type) type {
    return @Vector(lanes(T), T);
}
pub fn Stream(comptime T: type) type {
    return extern struct {
        veks: [*]Vek(T),
        len_veks: usize,
        len_scalars: usize,
    };
}
pub const Ctx = struct {
    inner: std.heap.FixedBufferAllocator,
    allocator: Allocator,
    buffer: []u8,
};
var default_allocator: std.heap.GeneralPurposeAllocator(.{}) = .init;
var gpa = default_allocator.allocator();

pub export fn make(size: usize) *Ctx {
    const backing_mem = gpa.alloc(u8, size) catch unreachable;
    const fba = std.heap.FixedBufferAllocator.init(backing_mem);
    const ctx = gpa.create(Ctx) catch unreachable;
    ctx.* = Ctx{
        .inner = fba,
        .allocator = undefined,
        .buffer = backing_mem,
    };
    ctx.allocator = ctx.inner.allocator();
    return ctx;
}

pub export fn free(ctx: *Ctx) void {
    gpa.free(ctx.buffer);
    gpa.destroy(ctx);
}

pub fn to_stream(comptime T: type, ctx: *Ctx, slice: []T) Stream(T) {
    var stream = new_stream(T, ctx, slice.len);
    set_stream(T, &stream, slice);
    return stream;
}

pub fn new_stream(comptime T: type, ctx: *Ctx, len: usize) Stream(T) {
    const num_veks = ceildiv(len, lanes(T));
    const stream: []Vek(T) = ctx.allocator.alloc(Vek(T), num_veks) catch @panic("OOM trying to create new stream");
    return Stream(T){
        .veks = stream.ptr,
        .len_veks = num_veks,
        .len_scalars = len,
    };
}
pub fn set_stream(comptime T: type, stream: *Stream(T), slice: []T) void {
    assert(stream.len_scalars == slice.len);
    const stream_ptr: [*]T = @ptrCast(stream.veks);
    @memcpy(stream_ptr, slice);
}

pub fn Operand(comptime T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => Stream(T),
        .Number => T,
    };
}

fn to_slice(comptime T: type, s: Stream(T)) []T {
    var slice: [*]T = @ptrCast(s.veks);
    return slice[0..s.len_scalars];
}

pub fn apply(
    comptime T: type,
    comptime op: Simd_Op,
    c: *Stream(T),
    comptime a_t: Operand_Variant,
    a: Operand(T, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(T, b_t),
) void {
    if (a_t == .Number and b_t == .Number) {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    if (a_t == .Vector and b_t == .Vector) {
        assert(a.len_veks == b.len_veks and b.len_veks == c.len_veks);
    }

    for (0..c.len_veks) |i| {
        const a_vek: Vek(T) = switch (a_t) {
            .Vector => a.veks[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Vector => b.veks[i],
            .Number => @splat(b),
        };
        c.veks[i] = switch (op) {
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
    c: *Stream(bool),
    comptime a_t: Operand_Variant,
    a: Operand(T, a_t),
    comptime b_t: Operand_Variant,
    b: Operand(T, b_t),
) void {
    if (a_t == .Number and b_t == .Number) {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    if (a_t == .Vector and b_t == .Vector) {
        assert(a.len_veks == b.len_veks and b.len_veks == c.len_veks);
    }

    const vek_start = c.start / lanes(T);
    const vek_end = (c.stop - 1) / lanes(T) + 1;

    for (vek_start..vek_end) |i| {
        const a_vek: Vek(T) = switch (a_t) {
            .Vector => a.veks[i],
            .Number => @splat(a),
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Vector => b.veks[i],
            .Number => @splat(b),
        };
        c.veks[i] = switch (Op) {
            .Gt => a_vek > b_vek,
            .Gte => a_vek >= b_vek,
            .Lt => a_vek < b_vek,
            .Lte => a_vek <= b_vek,
            .Eq => a_vek == b_vek,
            .Neq => a_vek != b_vek,
        };
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
        c.veks[i] = switch (Op) {
            .Ceil => @ceil(c.veks[i]),
            .Floor => @floor(c.veks[i]),
            .Round => @round(c.veks[i]),
            .Sqrt => @sqrt(c.veks[i]),
            .Not => ~c.veks[i],
            .Neg => -c.veks[i],
        };
    }
}

// TODO
pub const SimdOp3 = enum {
    Fma,
};

test "basic add" {
    const num_t = f64;

    const ctx = make(1024);

    var a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const a_stream = to_stream(num_t, ctx, &a_arr);

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const b_stream = to_stream(num_t, ctx, &b_arr);

    var c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    const c_expect_stream = to_stream(num_t, ctx, &c_expect_arr);

    var c_actual_stream = new_stream(num_t, ctx, b_stream.len_scalars);
    apply(num_t, .Add, &c_actual_stream, .Vector, a_stream, .Vector, b_stream);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    try testing.expectEqualSlices(num_t, to_slice(num_t, c_expect_stream), to_slice(num_t, c_actual_stream));
}

comptime {
    @setEvalBranchQuota(4096);
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
            generate_apply_func(t, op_enum, .Vector, .Vector);
            generate_apply_func(t, op_enum, .Vector, .Number);
            generate_apply_func(t, op_enum, .Number, .Vector);
        }
    }
}

fn generate_apply_func(
    comptime T: type,
    comptime op: Simd_Op,
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
    @export(&struct {
        fn generated(c: *Stream(T), a: Operand(T, a_t), b: Operand(T, b_t)) callconv(.C) void {
            apply(T, op, c, a_t, a, b_t, b);
        }
    }.generated, .{ .name = name });
}
