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

pub fn Operand(comptime T: type, comptime variant: Operand_Variant) type {
    return switch (variant) {
        .Vector => [*]T,
        .Number => T,
    };
}

fn Apply_Args(comptime T: type, comptime a_t: Operand_Variant, b_t: Operand_Variant) type {
    return extern struct {
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
    const simd_len = (len / num_lanes) * num_lanes;

    var i: usize = 0;
    while (i < simd_len) : (i += num_lanes) {
        const a_vek: Vek(T) = switch (a_t) {
            .Number => @splat(a),
            .Vector => a[i..][0..num_lanes].*,
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Number => @splat(b),
            .Vector => b[i..][0..num_lanes].*,
        };
        c[i..][0..num_lanes].* = switch (op) {
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
    //leftovers
    while (i < len) : (i += 1) {
        const a_num: T = switch (a_t) {
            .Number => a,
            .Vector => a[i],
        };
        const b_num: T = switch (b_t) {
            .Number => b,
            .Vector => b[i],
        };
        c[i] = switch (op) {
            .Add => a_num + b_num,
            .Sub => a_num - b_num,
            .Div => @divTrunc(a_num, b_num),
            .Mul => a_num * b_num,
            .Mod => @mod(a_num, b_num),
            .Min => @min(a_num, b_num),
            .Max => @max(a_num, b_num),
            .Log => @log2(b_num) / @log2(a_num),
            .Pow => @exp2(b_num * @log2(a_num)),
            // .LShift => a_num << b_num,
            // .RShift => a_num >> b_num,
            // .SLShift => a_num <<| b_num,
            .And => a_num & b_num,
            .Or => a_num | b_num,
            .Xor => a_num ^ b_num,
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
        c: [*]bool,
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
    const simd_len = (len / num_lanes) * num_lanes;

    var i: usize = 0;
    while (i < simd_len) : (i += num_lanes) {
        const a_vek: Vek(T) = switch (a_t) {
            .Number => @splat(a),
            .Vector => a[i..][0..num_lanes].*,
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Number => @splat(b),
            .Vector => b[i..][0..num_lanes].*,
        };
        c[i..][0..num_lanes].* = switch (op) {
            .Gt => a_vek > b_vek,
            .Gte => a_vek >= b_vek,
            .Lt => a_vek < b_vek,
            .Lte => a_vek <= b_vek,
            .Eq => a_vek == b_vek,
            .Neq => a_vek != b_vek,
        };
    }
    // leftovers
    while (i < len) : (i += 1) {
        const a_num: T = switch (a_t) {
            .Number => a,
            .Vector => a[i],
        };
        const b_num: T = switch (b_t) {
            .Number => b,
            .Vector => b[i],
        };
        c[i] = switch (op) {
            .Gt => a_num > b_num,
            .Gte => a_num >= b_num,
            .Lt => a_num < b_num,
            .Lte => a_num <= b_num,
            .Eq => a_num == b_num,
            .Neq => a_num != b_num,
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
};

const signed_op_1s = [_]Simd_Op1{
    .Neg,
    .Abs,
};

const byte_op_1s = [_]Simd_Op1{.Not};

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
    const simd_len = (len / num_lanes) * num_lanes;
    const vek_t = @Vector(num_lanes, T);
    const vek_o = @Vector(num_lanes, O);
    var i: usize = 0;
    while (i < simd_len) : (i += num_lanes) {
        const x_vek: vek_t = x[i..][0..num_lanes].*;
        y[i..][0..num_lanes].* = switch (Op) {
            .Ceil => @ceil(x_vek),
            .Floor => @floor(x_vek),
            .Round => @round(x_vek),
            .Sqrt => @sqrt(x_vek),
            .Not => ~x_vek,
            .Neg => -x_vek,
            .Abs => @abs(x_vek),
            .Ln => @log(x_vek),
            .Log2 => @log2(x_vek),
            .Log10 => @log10(x_vek),
            .Exp => @exp(x_vek),
            .Exp2 => @exp2(x_vek),
            .Cast => cast(vek_t, vek_o, x_vek),
        };
    }
    // leftovers
    while (i < len) : (i += 1) {
        y[i] = switch (Op) {
            .Ceil => @ceil(x[i]),
            .Floor => @floor(x[i]),
            .Round => @round(x[i]),
            .Sqrt => @sqrt(x[i]),
            .Not => ~x[i],
            .Neg => -x[i],
            .Abs => @abs(x[i]),
            .Ln => @log(x[i]),
            .Log2 => @log2(x[i]),
            .Log10 => @log10(x[i]),
            .Exp => @exp(x[i]),
            .Exp2 => @exp2(x[i]),
            .Cast => cast(T, O, x[i]),
        };
    }
}

pub const Simd_Op_Cum = enum {
    CumSum,
    CumProd,
};

const simd_cum_ops = [_]Simd_Op_Cum{
    .CumSum,
    .CumProd,
};

fn Apply_Args_Cum(comptime T: type) type {
    return struct {
        x: [*]T,
        len: usize,
    };
}

pub inline fn apply_cum(
    comptime T: type,
    comptime Op: Simd_Op_Cum,
    args: Apply_Args_Cum(T),
) T {
    const x = args.x;
    const len = args.len;
    const num_lanes = lanes(T);
    const simd_len = (len / num_lanes) * num_lanes;
    const vek_t = Vek(T);
    var acc: vek_t = @splat(0);
    var i: usize = 0;
    while (i < simd_len) : (i += num_lanes) {
        const x_vek: vek_t = x[i..][0..num_lanes].*;
        acc = switch (Op) {
            .CumSum => acc + x_vek,
            .CumProd => acc * x_vek,
        };
    }
    var res: T = 0;
    // leftovers
    while (i < len) : (i += 1) {
        res = switch (Op) {
            .CumSum => res + x[i],
            .CumProd => res * x[i],
        };
    }
    const arr: [num_lanes]T = acc;
    for (arr) |value| {
        res = switch (Op) {
            .CumSum => res + value,
            .CumProd => res * value,
        };
    }
    return res;
}

// TODO
pub const SimdOp3 = enum {
    Fma,
};

fn Select_Args(comptime T: type, comptime a_t: Operand_Variant, b_t: Operand_Variant) type {
    return struct {
        a: Operand(T, a_t),
        b: Operand(T, b_t),
        pred: [*]bool,
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
    const simd_len = (len / num_lanes) * num_lanes;

    var i: usize = 0;
    while (i < simd_len) : (i += num_lanes) {
        const a_vek: Vek(T) = switch (a_t) {
            .Number => @splat(a),
            .Vector => a[i..][0..num_lanes].*,
        };
        const b_vek: Vek(T) = switch (b_t) {
            .Number => @splat(b),
            .Vector => b[i..][0..num_lanes].*,
        };
        const pred_vek: @Vector(num_lanes, bool) = pred[i..][0..num_lanes].*;
        c[i..][0..num_lanes].* = @select(T, pred_vek, a_vek, b_vek);
    }
    // leftovers
    while (i < len) : (i += 1) {
        const a_num: T = switch (a_t) {
            .Number => a,
            .Vector => a[i],
        };
        const b_num: T = switch (b_t) {
            .Number => b,
            .Vector => b[i],
        };
        c[i] = if (pred[i]) a_num else b_num;
    }
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
            if (t != u8) {
                for (byte_op_1s) |byte_op| {
                    if (op_enum == byte_op) {
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
        for (simd_cum_ops) |op| {
            generate_apply_cum_func(t, op);
        }
    }
}

fn Set_Args(comptime T: type) type {
    return struct {
        a: [*]T,
        b: T,
        len: usize,
    };
}

fn generate_set_func(comptime T: type) void {
    const name: []const u8 = "Set_" ++ @typeName(T);
    @export(&struct {
        fn generated(args: *Set_Args(T)) callconv(.C) void {
            @memset(args.a[0..args.len], args.b);
        }
    }.generated, .{ .name = name });
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
fn generate_apply_cum_func(
    comptime T: type,
    comptime op: Simd_Op_Cum,
) void {
    const name: []const u8 = @tagName(op) ++ "_" ++ @typeName(T);
    @export(&struct {
        fn generated(args: *Apply_Args_Cum(T)) callconv(.C) T {
            return apply_cum(T, op, args.*);
        }
    }.generated, .{ .name = name });
}

test "basic add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer _ = arena.reset(.free_all);
    const al = arena.allocator();
    const num_t = f64;

    // init();

    var a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };

    const c_actual = try al.alloc(num_t, b_arr.len);
    const args = Apply_Args(num_t, .Vector, .Vector){
        .a = &a_arr,
        .b = &b_arr,
        .c = c_actual.ptr,
        .len = a_arr.len,
    };
    apply(num_t, .Add, .Vector, .Vector, args);
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    const c_expect_arr = [_]num_t{ 0, 2, 10, 165, 1000, 1110, 16665, 100, 200, 20, 40 };
    try testing.expectEqualSlices(num_t, &c_expect_arr, c_actual);
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
    @setEvalBranchQuota(4096);
    var randy = std.Random.DefaultPrng.init(std.testing.random_seed);
    var r = randy.random();
    const buf = try std.testing.allocator.alloc(u8, 4096 * 3 * 16);
    defer std.testing.allocator.free(buf);
    var fba = std.heap.FixedBufferAllocator.init(buf);
    const al = fba.allocator();
    inline for (number_types) |in| {
        inline for (number_types) |out| {
            if (in == out) {
                continue;
            }
            defer fba.reset();
            const len = r.uintLessThan(usize, 4096);

            var a_stream = try al.alloc(in, len);
            var b_stream = try al.alloc(out, len);
            for (0..len) |i| {
                const value = smart_random(in, out, r);
                a_stream[i] = value;
                b_stream[i] = cast(in, out, value);
            }

            const b_actual_stream = try al.alloc(out, len);
            const args = Apply_Args_Single(in, out){
                .x = a_stream.ptr,
                .y = b_actual_stream.ptr,
                .len = len,
            };
            apply_single(in, out, .Cast, args);
            // print_vek(num_t, c_expect_stream);
            // print_vek(num_t, c_actual_stream);
            try testing.expectEqualSlices(out, b_stream, b_actual_stream);
        }
    }
}
test "apply bool + select" {
    const num_t = f64;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer _ = arena.reset(.free_all);
    const al = arena.allocator();

    var a_arr = [_]num_t{ 0, 1, 7, 69, 420, 666, 6969, 50, 100, 10, 20 };

    var b_arr = [_]num_t{ 0, 1, 3, 96, 580, 444, 9696, 50, 100, 10, 20 };

    const pred_stream = try al.alloc(bool, b_arr.len);
    apply_bool(num_t, .Eq, .Vector, .Vector, Apply_Bool_Args(num_t, .Vector, .Vector){
        .a = &a_arr,
        .b = &b_arr,
        .c = pred_stream.ptr,
        .len = a_arr.len,
    });

    const c_actual_stream = try al.alloc(num_t, b_arr.len);
    select(num_t, .Vector, .Number, Select_Args(num_t, .Vector, .Number){
        .a = &a_arr,
        .b = -1,
        .pred = pred_stream.ptr,
        .c = c_actual_stream.ptr,
        .len = b_arr.len,
    });
    // print_vek(num_t, c_expect_stream);
    // print_vek(num_t, c_actual_stream);
    var c_expect_arr = [_]num_t{ 0, 1, -1, -1, -1, -1, -1, 50, 100, 10, 20 };
    try testing.expectEqualSlices(num_t, &c_expect_arr, c_actual_stream);
}
