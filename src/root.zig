const std = @import("std");
const builtin = @import("builtin");
const testing = std.testing;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

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

const simd_byte_ops = [_]Simd_Op{
    .And,
    .Or,
    .Xor,
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
    i128,
    u128,
    f32,
    f64,
};

const float_types = [_]type{
    f32, f64,
};

const unsigned_types = [_]type{ u8, u16, u32, u64, u128 };

pub const Operand_Variant = enum {
    Vector,
    Number,
};

fn ceildiv(x: usize, y: usize) usize {
    return x / y + @intFromBool(x % y != 0);
}

fn print_float_slice(comptime T: type, s: []T) void {
    for (s) |x| {
        std.debug.print("{d:.2} ", .{x});
    }
    std.debug.print("\n", .{});
}

pub const TARGET_SIMD_BYTES: comptime_int = 64; // AVX512
pub fn lanes(comptime T: type) comptime_int {
    return TARGET_SIMD_BYTES / @sizeOf(T);
}

pub fn Vek(comptime T: type) type {
    return @Vector(lanes(T), T);
}
pub const Ctx = struct {
    inner: std.heap.FixedBufferAllocator,
    allocator: Allocator,
    buffer: []u8,
};

const al: Allocator = if (builtin.is_test) std.testing.allocator else std.heap.page_allocator;
var ctx_al: std.heap.MemoryPool(Ctx) = undefined;

pub export fn init() void {
    ctx_al = std.heap.MemoryPool(Ctx).init(al);
}

pub export fn deinit() void {
    ctx_al.deinit();
}

pub export fn make_ctx(size: usize) *Ctx {
    const ctx = ctx_al.create() catch unreachable;
    const ctx_backing_mem = al.alloc(u8, size) catch unreachable;
    const ctx_fba = std.heap.FixedBufferAllocator.init(ctx_backing_mem);
    ctx.* = Ctx{
        .inner = ctx_fba,
        .allocator = undefined,
        .buffer = ctx_backing_mem,
    };
    ctx.allocator = ctx.inner.allocator();
    return ctx;
}

pub export fn reset_ctx(ctx: *Ctx) void {
    ctx.inner.reset();
}

pub export fn free_ctx(ctx: *Ctx) void {
    al.free(ctx.buffer);
    ctx_al.destroy(ctx);
}

pub fn make_stream(comptime T: type, ctx: *Ctx, len: usize) [*]T {
    const num_lanes = lanes(T);
    const internal_size = ceildiv(len, num_lanes) * num_lanes;
    const stweam = ctx.allocator.alignedAlloc(T, TARGET_SIMD_BYTES, internal_size) catch @panic("OOM trying to create new stream");
    @memset(stweam, 0);
    return stweam.ptr;
}

pub fn Operand(comptime T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => [*]T,
        .Number => T,
    };
}

fn Apply_Args(comptime T: type, comptime a_t: Operand_Variant, b_t: Operand_Variant) type {
    return struct {
        a: Operand(T, a_t),
        b: Operand(T, b_t),
        c: [*]T,
        len: usize,
    };
}

pub inline fn apply(
    comptime T: type,
    comptime op: Simd_Op,
    comptime a_t: Operand_Variant,
    comptime b_t: Operand_Variant,
    args: Apply_Args(T, a_t, b_t),
) void {
    if (a_t == .Number and b_t == .Number) {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    const num_lanes = lanes(T);
    const a = args.a;
    const b = args.b;
    const c = args.c;
    const len = args.len;

    // streams are guarunteed to be 64 bytes min

    var a_vek: Vek(T) = if (a_t == .Number) @splat(a) else undefined;
    var a_vek_ptr: *Vek(T) = &a_vek;
    var b_vek: Vek(T) = if (b_t == .Number) @splat(b) else undefined;
    var b_vek_ptr: *Vek(T) = &b_vek;
    var i: usize = 0;
    while (i < len) : (i += num_lanes) {
        if (a_t == .Vector) {
            a_vek_ptr = @ptrCast(@alignCast(a[i..][0..num_lanes]));
            a_vek = a_vek_ptr.*;
        }
        if (b_t == .Vector) {
            b_vek_ptr = @ptrCast(@alignCast(b[i..][0..num_lanes]));
            b_vek = b_vek_ptr.*;
        }
        const c_vek_ptr: *Vek(T) = @ptrCast(@alignCast(c[i..][0..num_lanes]));
        c_vek_ptr.* = switch (op) {
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
            .And => a_vek & b_vek,
            .Or => a_vek | b_vek,
            .Xor => a_vek ^ b_vek,
        };
    }
}

pub const Simd_Bool_Op = enum {
    Gt,
    Gte,
    Lt,
    Lte,
    Eq,
    Neq,
};

fn Apply_Bool_Args(comptime T: type, comptime a_t: Operand_Variant, b_t: Operand_Variant) type {
    return struct {
        a: Operand(T, a_t),
        b: Operand(T, b_t),
        c: [*]u1,
        len: usize,
    };
}

pub inline fn apply_bool(
    comptime T: type,
    comptime op: Simd_Bool_Op,
    comptime a_t: Operand_Variant,
    comptime b_t: Operand_Variant,
    args: Apply_Bool_Args(T, a_t, b_t),
) void {
    if (a_t == .Number and b_t == .Number) {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    const a = args.a;
    const b = args.b;
    const c = args.c;
    const len = args.len;
    const num_lanes = lanes(T);
    var a_vek: Vek(T) = if (a_t == .Number) @splat(a) else undefined;
    var a_vek_ptr: *Vek(T) = &a_vek;
    var b_vek: Vek(T) = if (b_t == .Number) @splat(b) else undefined;
    var b_vek_ptr: *Vek(T) = &b_vek;
    var i: usize = 0;
    while (i < len) : (i += num_lanes) {
        if (a_t == .Vector) {
            a_vek_ptr = @ptrCast(@alignCast(a[i..][0..num_lanes]));
            a_vek = a_vek_ptr.*;
        }
        if (b_t == .Vector) {
            b_vek_ptr = @ptrCast(@alignCast(b[i..][0..num_lanes]));
            b_vek = b_vek_ptr.*;
        }
        const c_vek_ptr: *@Vector(num_lanes, bool) = @ptrCast(@alignCast(c[i..][0..num_lanes]));
        c_vek_ptr.* = switch (op) {
            .Gt => a_vek > b_vek,
            .Gte => a_vek >= b_vek,
            .Lt => a_vek < b_vek,
            .Lte => a_vek <= b_vek,
            .Eq => a_vek == b_vek,
            .Neq => a_vek != b_vek,
        };
    }
}

pub const Simd_Op1 = enum {
    Ceil,
    Floor,
    Round,
    Sqrt,
    Not,
    Neg,
    Abs,
    Ln,
    Log2,
    Log10,
    Exp,
    Exp2,
    Cast,
};

const simd_float_op_1s = [_]Simd_Op1{
    .Ln,
    .Log2,
    .Log10,
    .Exp,
    .Exp2,
    .Sqrt,
    .Ceil,
    .Floor,
    .Round,
    .Cast,
};

const signed_op_1s = [_]Simd_Op1{
    .Neg,
    .Abs,
};

fn unsigned_variant(comptime T: type) type {
    return switch (T) {
        i8 => u8,
        i16 => u16,
        i32 => u32,
        i64 => u64,
        i128 => u128,
        f32 => f32,
        f64 => f64,
        else => comptime unreachable,
    };
}

fn Apply_Args_Single(comptime T: type, comptime O: type) type {
    return struct {
        x: [*]T,
        y: [*]O,
        len: usize,
    };
}

fn int_by_size(comptime signed: bool, size: comptime_int) type {
    return switch (signed) {
        true => switch (size) {
            1 => i8,
            2 => i16,
            4 => i32,
            8 => i64,
            else => comptime @compileError(std.fmt.comptimePrint("signed {} byte integer no existy my friend", .{size})),
        },
        false => switch (size) {
            1 => u8,
            2 => u16,
            4 => u32,
            8 => u64,
            else => comptime @compileError(std.fmt.comptimePrint("unsigned {} byte integer no existy my friend", .{size})),
        },
    };
}
fn vector_int_by_size(comptime signed: bool, len_child: comptime_int, size: comptime_int) type {
    const scalar_t = int_by_size(signed, size);
    return @Vector(len_child, scalar_t);
}

pub fn cast(comptime T: type, comptime O: type, value: T) O {
    comptime var in_type = @typeInfo(T);
    comptime var out_type = @typeInfo(O);

    if (in_type == .vector and out_type == .vector) {
        in_type = @typeInfo(in_type.vector.child);
        out_type = @typeInfo(out_type.vector.child);
    }
    if (in_type == .optional) {
        return cast(T, O, value.?);
    }
    return switch (out_type) {
        .int => switch (in_type) {
            .int => @intCast(value),
            .float => @intFromFloat(value),
            .bool => @intFromBool(value),
            .@"enum" => @intFromEnum(value),
            .pointer => @intFromPtr(value),
            else => invalid(T, O),
        },
        .float => switch (in_type) {
            .int => @floatFromInt(value),
            .float => @floatCast(value),
            .bool => @floatFromInt(@intFromBool(value)),
            else => invalid(T, O),
        },
        .bool => switch (in_type) {
            .int => value != 0,
            .float => value != 0,
            .bool => value,
            else => invalid(T, O),
        },
        .@"enum" => switch (in_type) {
            .int => @enumFromInt(value),
            .@"enum" => @enumFromInt(@intFromEnum(value)),
            else => invalid(T, O),
        },
        .pointer => switch (in_type) {
            .int => @ptrFromInt(value),
            .pointer => @ptrCast(value),
            else => invalid(T, O),
        },
        else => invalid(T, O),
    };
}

pub fn invalid(comptime in: type, comptime out: type) noreturn {
    @compileError("cast: " ++ @typeName(in) ++ " to " ++ @typeName(out) ++ " not supported");
}

pub inline fn apply_single(
    comptime T: type,
    comptime O: type,
    comptime Op: Simd_Op1,
    args: Apply_Args_Single(T, O),
) void {
    const x = args.x;
    const y = args.y;
    const len = args.len;
    const num_lanes = @min(lanes(T), lanes(O));
    const vek_t = @Vector(num_lanes, T);
    const vek_o = @Vector(num_lanes, O);
    var i: usize = 0;
    while (i < len) : (i += num_lanes) {
        const x_ptr: *vek_t = @ptrCast(@alignCast(x[i..][0..num_lanes]));
        const y_ptr: *vek_o = @ptrCast(@alignCast(y[i..][0..num_lanes]));
        y_ptr.* = switch (Op) {
            .Ceil => @ceil(x_ptr.*),
            .Floor => @floor(x_ptr.*),
            .Round => @round(x_ptr.*),
            .Sqrt => @sqrt(x_ptr.*),
            .Not => blk: {
                var c_bytes: @Vector(TARGET_SIMD_BYTES, u8) = @bitCast(x_ptr.*);
                c_bytes = switch (Op) {
                    .Not => ~c_bytes,
                    else => comptime unreachable,
                };
                const c_vek: vek_o = @bitCast(c_bytes);
                break :blk c_vek;
            },
            .Neg => -x_ptr.*,
            .Abs => @abs(x_ptr.*),
            .Ln => @log(x_ptr.*),
            .Log2 => @log2(x_ptr.*),
            .Log10 => @log10(x_ptr.*),
            .Exp => @exp(x_ptr.*),
            .Exp2 => @exp2(x_ptr.*),
            .Cast => cast(vek_t, vek_o, x_ptr.*),
        };
    }
}

// TODO
pub const SimdOp3 = enum {
    Fma,
};

fn Select_Args(comptime T: type, comptime a_t: Operand_Variant, b_t: Operand_Variant) type {
    return struct {
        a: Operand(T, a_t),
        b: Operand(T, b_t),
        pred: [*]u1,
        c: [*]T,
        len: usize,
    };
}

pub inline fn select(comptime T: type, comptime a_t: Operand_Variant, comptime b_t: Operand_Variant, args: Select_Args(T, a_t, b_t)) void {
    if (a_t == .Number and b_t == .Number) {
        @compileError("Adding scalars together should not be done in a vectorized operation bro");
    }
    const a = args.a;
    const b = args.b;
    const c = args.c;
    const len = args.len;
    const pred = args.pred;
    const num_lanes = lanes(T);

    var a_vek: Vek(T) = if (a_t == .Number) @splat(a) else undefined;
    var a_vek_ptr: *Vek(T) = &a_vek;
    var b_vek: Vek(T) = if (b_t == .Number) @splat(b) else undefined;
    var b_vek_ptr: *Vek(T) = &b_vek;
    var i: usize = 0;
    while (i < len) : (i += num_lanes) {
        if (a_t == .Vector) {
            a_vek_ptr = @ptrCast(@alignCast(a[i..][0..num_lanes]));
            a_vek = a_vek_ptr.*;
        }
        if (b_t == .Vector) {
            b_vek_ptr = @ptrCast(@alignCast(b[i..][0..num_lanes]));
            b_vek = b_vek_ptr.*;
        }
        const c_vek_ptr: *Vek(T) = @ptrCast(@alignCast(c[i..][0..num_lanes]));
        const pred_vek: *@Vector(num_lanes, bool) = @ptrCast(@alignCast(pred[i..][0..num_lanes]));
        c_vek_ptr.* = @select(T, pred_vek.*, a_vek, b_vek);
    }
}

comptime {
    @setEvalBranchQuota(4096);
    for (number_types) |t| {
        generate_new_stream_func(t);

        var is_float_t = false;
        for (float_types) |float_ts| {
            if (t != float_ts) continue;
            is_float_t = true;
            break;
        }
        var is_signed_type = true;
        for (unsigned_types) |u_type| {
            if (t != u_type) continue;
            is_signed_type = false;
            break;
        }
        // normal apply
        normal_loop: for (@typeInfo(Simd_Op).@"enum".fields) |op| {
            const op_enum: Simd_Op = @enumFromInt(op.value);
            if (!is_float_t) {
                for (simd_float_ops) |float_op| {
                    if (op_enum == float_op) {
                        continue :normal_loop;
                    }
                }
            }
            if (t != u8) {
                for (simd_byte_ops) |byte_op| {
                    if (op_enum == byte_op) {
                        continue :normal_loop;
                    }
                }
            }
            generate_apply_func(t, op_enum, .Vector, .Vector);
            generate_apply_func(t, op_enum, .Vector, .Number);
            generate_apply_func(t, op_enum, .Number, .Vector);
        }
        // select
        generate_select_func(t, .Vector, .Vector);
        generate_select_func(t, .Number, .Vector);
        generate_select_func(t, .Vector, .Number);
        // bool
        for (@typeInfo(Simd_Bool_Op).@"enum".fields) |op| {
            const op_enum: Simd_Bool_Op = @enumFromInt(op.value);
            generate_apply_bool_func(t, op_enum, .Vector, .Vector);
            generate_apply_bool_func(t, op_enum, .Vector, .Number);
            generate_apply_bool_func(t, op_enum, .Number, .Vector);
        }
        // single apply
        single_loop: for (@typeInfo(Simd_Op1).@"enum".fields) |op| {
            const op_enum: Simd_Op1 = @enumFromInt(op.value);
            if (op_enum == .Cast) {
                continue :single_loop;
            }
            if (!is_float_t) {
                for (simd_float_op_1s) |float_op| {
                    if (op_enum == float_op) {
                        continue :single_loop;
                    }
                }
            }
            if (!is_signed_type) {
                for (signed_op_1s) |signed_op| {
                    if (op_enum == signed_op) {
                        continue :single_loop;
                    }
                }
            }
            const out_t = switch (op_enum) {
                .Abs => unsigned_variant(t),
                else => t,
            };
            generate_apply_single_func(t, out_t, op_enum);
        }
        for (number_types) |cast_t| {
            if (cast_t == t) {
                continue;
            }
            generate_apply_single_func(t, cast_t, .Cast);
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
        name = "Num_" ++ name;
    }
    if (b_t == .Number) {
        name = name ++ "_Num";
    }
    @export(&struct {
        fn generated(args: *Apply_Args(T, a_t, b_t)) callconv(.C) void {
            apply(T, op, a_t, b_t, args.*);
        }
    }.generated, .{ .name = name });
}

fn generate_select_func(
    comptime T: type,
    comptime a_t: Operand_Variant,
    comptime b_t: Operand_Variant,
) void {
    var name: []const u8 = "Select_" ++ @typeName(T);
    if (a_t == .Number) {
        name = "Num_" ++ name;
    }
    if (b_t == .Number) {
        name = name ++ "_Num";
    }
    @export(&struct {
        fn generated(args: *Select_Args(T, a_t, b_t)) callconv(.C) void {
            select(T, a_t, b_t, args.*);
        }
    }.generated, .{ .name = name });
}

fn generate_new_stream_func(comptime T: type) void {
    const name: []const u8 = "New_" ++ @typeName(T) ++ "_Stream";
    @export(&struct {
        fn generated(ctx: *Ctx, len: usize) callconv(.C) [*]T {
            return make_stream(T, ctx, len);
        }
    }.generated, .{ .name = name });
}

fn generate_apply_bool_func(
    comptime T: type,
    comptime op: Simd_Bool_Op,
    comptime a_t: Operand_Variant,
    comptime b_t: Operand_Variant,
) void {
    var name: []const u8 = @tagName(op) ++ "_" ++ @typeName(T);
    if (a_t == .Number) {
        name = "Num_" ++ name;
    }
    if (b_t == .Number) {
        name = name ++ "_Num";
    }
    @export(&struct {
        fn generated(args: *Apply_Bool_Args(T, a_t, b_t)) callconv(.C) void {
            apply_bool(T, op, a_t, b_t, args.*);
        }
    }.generated, .{ .name = name });
}
fn generate_apply_single_func(
    comptime T: type,
    comptime O: type,
    comptime op: Simd_Op1,
) void {
    var name: []const u8 = @tagName(op) ++ "_" ++ @typeName(T);
    if (T != O) {
        name = name ++ "_" ++ @typeName(O);
    }
    @export(&struct {
        fn generated(args: *Apply_Args_Single(T, O)) callconv(.C) void {
            apply_single(T, O, op, args.*);
        }
    }.generated, .{ .name = name });
}

test "basic add" {
    init();
    defer deinit();
    const num_t = f64;

    // init();
    const ctx = make_ctx(1024);
    defer free_ctx(ctx);

    const a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const a_stream = make_stream(num_t, ctx, a_arr.len);
    @memcpy(a_stream, &a_arr);

    const b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const b_stream = make_stream(num_t, ctx, b_arr.len);
    @memcpy(b_stream, &b_arr);

    const c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    const c_expect_stream = make_stream(num_t, ctx, c_expect_arr.len);
    @memcpy(c_expect_stream, &c_expect_arr);

    const c_actual_stream = make_stream(num_t, ctx, b_arr.len);
    const args = Apply_Args(num_t, .Vector, .Vector){
        .a = a_stream,
        .b = b_stream,
        .c = c_actual_stream,
        .len = a_arr.len,
    };
    apply(num_t, .Add, .Vector, .Vector, args);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    try testing.expectEqualSlices(num_t, c_expect_stream[0..c_expect_arr.len], c_actual_stream[0..b_arr.len]);
}

fn smart_random(comptime T: type, comptime O: type, r: std.Random) T {
    const t_info = @typeInfo(T);
    const o_info = @typeInfo(O);
    return switch (t_info) {
        .int => blk: {
            const max_val = switch (o_info) {
                .int => @min(std.math.maxInt(T), std.math.maxInt(O)),
                .float => @min(std.math.maxInt(T), std.math.maxInt(i32)),
                else => comptime unreachable,
            };
            const min_val = switch (o_info) {
                .int => switch (o_info.int.signedness) {
                    .signed => @max(std.math.minInt(T), std.math.minInt(O)),
                    .unsigned => 0,
                },
                .float => @max(std.math.minInt(T), std.math.minInt(i32)),
                else => comptime unreachable,
            };
            break :blk switch (t_info.int.signedness) {
                .signed => r.intRangeLessThan(T, min_val, max_val),
                .unsigned => r.uintLessThan(T, max_val),
            };
        },
        .float => blk: {
            const max_val = switch (o_info) {
                .int => @min(std.math.maxInt(O), std.math.maxInt(i32)),
                .float => std.math.maxInt(i32),
                else => comptime unreachable,
            };
            const min_val = switch (o_info) {
                .int => switch (o_info.int.signedness) {
                    .signed => @max(std.math.minInt(O), std.math.minInt(i32)),
                    .unsigned => 0,
                },
                .float => std.math.minInt(i32),
                else => comptime unreachable,
            };
            const flert: T = @floatFromInt(r.intRangeLessThan(i32, min_val, max_val));
            break :blk r.float(T) * flert;
        },
        else => comptime unreachable,
    };
}

test "cast testing" {
    init();
    defer deinit();
    @setEvalBranchQuota(4096);
    var randy = std.Random.DefaultPrng.init(std.testing.random_seed);
    var r = randy.random();
    const ctx = make_ctx(4096 * @sizeOf(i128) * 3);
    defer free_ctx(ctx);
    inline for (number_types) |in| {
        inline for (number_types) |out| {
            if (in == out) {
                continue;
            }
            defer reset_ctx(ctx);
            const len = r.uintLessThan(usize, 4096);

            var a_stream = make_stream(in, ctx, len);
            var b_stream = make_stream(out, ctx, len);
            for (0..len) |i| {
                const value = smart_random(in, out, r);
                a_stream[i] = value;
                b_stream[i] = cast(in, out, value);
            }

            const b_actual_stream = make_stream(out, ctx, len);
            const args = Apply_Args_Single(in, out){
                .x = a_stream,
                .y = b_actual_stream,
                .len = len,
            };
            apply_single(in, out, .Cast, args);
            // print_vek(num_t, c_expect_stream);
            // print_vek(num_t, c_actual_stream);
            try testing.expectEqualSlices(out, b_stream[0..len], b_actual_stream[0..len]);
        }
    }
}
test "apply bool + select" {
    init();
    defer deinit();
    const num_t = f64;

    const ctx = make_ctx(1024);
    defer free_ctx(ctx);

    const a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };
    const a_stream = make_stream(num_t, ctx, a_arr.len);
    @memcpy(a_stream, &a_arr);

    const b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };
    const b_stream = make_stream(num_t, ctx, b_arr.len);
    @memcpy(b_stream, &b_arr);

    const pred_stream = make_stream(u1, ctx, b_arr.len);
    apply_bool(num_t, .Eq, .Vector, .Vector, Apply_Bool_Args(num_t, .Vector, .Vector){
        .a = a_stream,
        .b = b_stream,
        .c = pred_stream,
        .len = a_arr.len,
    });

    const c_actual_stream = make_stream(num_t, ctx, b_arr.len);
    select(num_t, .Vector, .Number, Select_Args(num_t, .Vector, .Number){
        .a = a_stream,
        .b = -1,
        .pred = pred_stream,
        .c = c_actual_stream,
        .len = b_arr.len,
    });
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    var c_expect_arr = [_]num_t{ 0, 1, -1, -1, -1, -1, -1, 50, 100, 10, 20 };
    try testing.expectEqualSlices(num_t, &c_expect_arr, c_actual_stream[0..a_arr.len]);
}
