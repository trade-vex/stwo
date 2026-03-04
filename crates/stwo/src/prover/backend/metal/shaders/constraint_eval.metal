// ============================================================================
// CONSTRAINT EVALUATION VM
// ============================================================================

/// VM State for constraint evaluation bytecode interpreter.
struct VMState {
    // M31 operand stack (up to 64 values)
    uint32_t m31_stack[64];
    uint m31_sp;  // Stack pointer for M31 stack

    // QM31 operand stack (up to 32 values)
    QM31 qm31_stack[32];
    uint qm31_sp;  // Stack pointer for QM31 stack

    // Accumulator for constraints
    QM31 constraint_accum;
    uint constraint_idx;  // Current constraint index

    // Instruction pointer
    uint ip;

    // Current row being evaluated
    uint row_idx;
};

// Bytecode instruction opcodes
#define OP_LOAD_TRACE_M31    0x00
#define OP_LOAD_CONST_M31    0x01
#define OP_DUP_M31           0x02
#define OP_POP_M31           0x03
#define OP_SWAP_M31          0x04

#define OP_ADD_M31           0x10
#define OP_SUB_M31           0x11
#define OP_MUL_M31           0x12
#define OP_NEG_M31           0x13
#define OP_SQUARE_M31        0x14
#define OP_INV_M31           0x15

#define OP_LOAD_TRACE_QM31   0x20
#define OP_LOAD_CONST_QM31   0x21
#define OP_DUP_QM31          0x22
#define OP_POP_QM31          0x23
#define OP_SWAP_QM31         0x24
#define OP_LOAD_RANDOM_COEFF 0x25

#define OP_ADD_QM31          0x30
#define OP_SUB_QM31          0x31
#define OP_MUL_QM31          0x32
#define OP_NEG_QM31          0x33
#define OP_SQUARE_QM31       0x34
#define OP_INV_QM31          0x35

#define OP_M31_TO_QM31       0x40
#define OP_COMBINE_EF        0x41
#define OP_MUL_M31_QM31      0x42
#define OP_ADD_M31_QM31      0x43

#define OP_ADD_CONSTRAINT    0x50
#define OP_MARK_INTERMEDIATE 0x51

#define OP_JUMP_IF_ZERO      0x60
#define OP_JUMP              0x61

#define OP_WRITE_LOGUP_FRAC  0x70
#define OP_FINALIZE_LOGUP    0x71

#define OP_PROGRAM_END       0xFF

/// Helper function to read big-endian u16 from bytecode
inline uint16_t read_u16_be(device const uint8_t* bytecode, thread uint* ip) {
    uint16_t result = (uint16_t(bytecode[*ip]) << 8) | uint16_t(bytecode[*ip + 1]);
    *ip += 2;
    return result;
}

/// Helper function to read big-endian i16 from bytecode
inline int16_t read_i16_be(device const uint8_t* bytecode, thread uint* ip) {
    uint16_t unsigned_val = read_u16_be(bytecode, ip);
    return int16_t(unsigned_val);
}

/// Helper function to read big-endian u32 from bytecode
inline uint32_t read_u32_be(device const uint8_t* bytecode, thread uint* ip) {
    uint32_t result = (uint32_t(bytecode[*ip]) << 24) |
                      (uint32_t(bytecode[*ip + 1]) << 16) |
                      (uint32_t(bytecode[*ip + 2]) << 8) |
                      uint32_t(bytecode[*ip + 3]);
    *ip += 4;
    return result;
}

/// Load QM31 from 4 consecutive u32 values
inline QM31 load_qm31(device const uint32_t* data) {
    QM31 result;
    result.c0.a = data[0];
    result.c0.b = data[1];
    result.c1.a = data[2];
    result.c1.b = data[3];
    return result;
}

/// Store QM31 to 4 consecutive u32 values
inline void store_qm31(device uint32_t* data, QM31 value) {
    data[0] = value.c0.a;
    data[1] = value.c0.b;
    data[2] = value.c1.a;
    data[3] = value.c1.b;
}

/// QM31 negate operation
inline QM31 qm31_neg(QM31 x) {
    QM31 result;
    result.c0.a = m31_sub(0, x.c0.a);
    result.c0.b = m31_sub(0, x.c0.b);
    result.c1.a = m31_sub(0, x.c1.a);
    result.c1.b = m31_sub(0, x.c1.b);
    return result;
}

/// QM31 square operation
inline QM31 qm31_square(QM31 x) {
    return qm31_mul(x, x);
}

/// CM31 negate operation
inline CM31 cm31_neg(CM31 x) {
    CM31 result;
    result.a = m31_sub(0, x.a);
    result.b = m31_sub(0, x.b);
    return result;
}

/// CM31 square operation
inline CM31 cm31_square(CM31 x) {
    return cm31_mul(x, x);
}

/// QM31 inverse operation
inline QM31 qm31_inv(QM31 x) {
    // (a+bi+cu+dui)^-1 where u^2 = 2+i
    // First get norm in CM31: (c0^2 - 2*c1*conj(c1) - i*|c1|^2)
    CM31 c0_sq = cm31_mul(x.c0, x.c0);
    CM31 c1_conj = {x.c1.a, m31_sub(0, x.c1.b)};
    CM31 c1_norm_sq = cm31_mul(x.c1, c1_conj);

    // 2*c1*conj(c1)
    CM31 two_c1_conj = cm31_add(cm31_mul(x.c1, c1_conj), cm31_mul(x.c1, c1_conj));

    // norm = c0^2 - 2*c1*conj(c1) - i*|c1|^2
    CM31 norm = cm31_sub(c0_sq, two_c1_conj);
    norm = cm31_sub(norm, cm31_mul({0, 1}, c1_norm_sq));

    // Inverse of norm
    CM31 norm_inv = cm31_inverse(norm);

    // Result = conj(x) / norm
    QM31 conj;
    conj.c0 = {x.c0.a, m31_sub(0, x.c0.b)};
    conj.c1 = cm31_neg({x.c1.a, m31_sub(0, x.c1.b)});

    QM31 result;
    result.c0 = cm31_mul(conj.c0, norm_inv);
    result.c1 = cm31_mul(conj.c1, norm_inv);
    return result;
}

/// Main constraint evaluation VM kernel
/// Evaluates constraint bytecode for a single row in the trace.
///
/// Parameters:
/// - bytecode: The bytecode program to execute
/// - bytecode_len: Length of bytecode in bytes
/// - trace: Flattened trace columns (column-major layout)
/// - column_offsets: Offset for each interaction's columns in trace
/// - n_columns: Number of columns per interaction
/// - random_coeffs: Random coefficients for constraint accumulation
/// - n_random_coeffs: Number of random coefficients
/// - denom_inv: Precomputed 1/coset_vanishing for each row
/// - trace_log_size: Log size of trace domain
/// - eval_log_size: Log size of evaluation domain
/// - output: Output buffer for constraint evaluation results
kernel void constraint_eval_vm(
    device const uint8_t* bytecode [[buffer(0)]],
    constant uint32_t& bytecode_len [[buffer(1)]],
    device const uint32_t* trace [[buffer(2)]],
    constant uint32_t* column_offsets [[buffer(3)]],
    constant uint32_t& n_columns [[buffer(4)]],
    device const uint32_t* random_coeffs [[buffer(5)]],
    constant uint32_t& n_random_coeffs [[buffer(6)]],
    device const uint32_t* denom_inv [[buffer(7)]],
    constant uint32_t& trace_log_size [[buffer(8)]],
    constant uint32_t& eval_log_size [[buffer(9)]],
    device uint32_t* output [[buffer(10)]],
    uint row_idx [[thread_position_in_grid]]
) {
    uint eval_domain_size = 1u << eval_log_size;

    // Bounds check
    if (row_idx >= eval_domain_size) {
        return;
    }

    // Initialize VM state
    VMState vm;
    vm.m31_sp = 0;
    vm.qm31_sp = 0;
    vm.constraint_accum = qm31_zero();
    vm.constraint_idx = 0;
    vm.ip = 0;
    vm.row_idx = row_idx;

    // Main interpreter loop
    while (vm.ip < bytecode_len) {
        uint8_t opcode = bytecode[vm.ip++];

        switch (opcode) {
            case OP_LOAD_TRACE_M31: {
                uint8_t interaction = bytecode[vm.ip++];
                uint16_t col_idx = read_u16_be(bytecode, &vm.ip);
                int16_t offset = read_i16_be(bytecode, &vm.ip);

                // Calculate actual column index
                uint actual_col = column_offsets[interaction] + col_idx;

                // Calculate row with offset wrapping
                int64_t offset_row = int64_t(row_idx) + int64_t(offset);

                // Wrap around eval domain (trace is pre-extended to eval domain)
                if (offset_row < 0) {
                    offset_row += eval_domain_size;
                } else if (offset_row >= eval_domain_size) {
                    offset_row -= eval_domain_size;
                }

                // Load value from trace
                uint32_t val = trace[actual_col * eval_domain_size + uint(offset_row)];
                vm.m31_stack[vm.m31_sp++] = val;
                break;
            }

            case OP_LOAD_CONST_M31: {
                uint32_t value = read_u32_be(bytecode, &vm.ip);
                vm.m31_stack[vm.m31_sp++] = value;
                break;
            }

            case OP_DUP_M31: {
                uint8_t depth = bytecode[vm.ip++];
                uint32_t val = vm.m31_stack[vm.m31_sp - 1 - depth];
                vm.m31_stack[vm.m31_sp++] = val;
                break;
            }

            case OP_POP_M31: {
                vm.m31_sp--;
                break;
            }

            case OP_SWAP_M31: {
                uint8_t depth1 = bytecode[vm.ip++];
                uint8_t depth2 = bytecode[vm.ip++];
                uint32_t temp = vm.m31_stack[vm.m31_sp - 1 - depth1];
                vm.m31_stack[vm.m31_sp - 1 - depth1] = vm.m31_stack[vm.m31_sp - 1 - depth2];
                vm.m31_stack[vm.m31_sp - 1 - depth2] = temp;
                break;
            }

            // M31 Arithmetic Operations
            case OP_ADD_M31: {
                uint32_t b = vm.m31_stack[--vm.m31_sp];
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_add(a, b);
                break;
            }

            case OP_SUB_M31: {
                uint32_t b = vm.m31_stack[--vm.m31_sp];
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_sub(a, b);
                break;
            }

            case OP_MUL_M31: {
                uint32_t b = vm.m31_stack[--vm.m31_sp];
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_mul(a, b);
                break;
            }

            case OP_NEG_M31: {
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_sub(0, a);
                break;
            }

            case OP_SQUARE_M31: {
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_square(a);
                break;
            }

            case OP_INV_M31: {
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                vm.m31_stack[vm.m31_sp++] = m31_inverse(a);
                break;
            }

            // QM31 Stack Operations
            case OP_LOAD_TRACE_QM31: {
                uint8_t interaction = bytecode[vm.ip++];
                uint16_t col_idx = read_u16_be(bytecode, &vm.ip);
                int16_t offset = read_i16_be(bytecode, &vm.ip);

                // Calculate actual column index (QM31 uses 4 consecutive M31 columns)
                uint actual_col = column_offsets[interaction] + col_idx;

                // Calculate row with offset wrapping
                int64_t offset_row = int64_t(row_idx) + int64_t(offset);
                if (offset_row < 0) {
                    offset_row += eval_domain_size;
                } else if (offset_row >= eval_domain_size) {
                    offset_row -= eval_domain_size;
                }

                // Load QM31 from 4 consecutive trace values
                QM31 val;
                uint row_offset = actual_col * eval_domain_size + uint(offset_row);
                val.c0.a = trace[row_offset];
                val.c0.b = trace[(actual_col + 1) * eval_domain_size + uint(offset_row)];
                val.c1.a = trace[(actual_col + 2) * eval_domain_size + uint(offset_row)];
                val.c1.b = trace[(actual_col + 3) * eval_domain_size + uint(offset_row)];
                vm.qm31_stack[vm.qm31_sp++] = val;
                break;
            }

            case OP_LOAD_CONST_QM31: {
                QM31 val;
                val.c0.a = read_u32_be(bytecode, &vm.ip);
                val.c0.b = read_u32_be(bytecode, &vm.ip);
                val.c1.a = read_u32_be(bytecode, &vm.ip);
                val.c1.b = read_u32_be(bytecode, &vm.ip);
                vm.qm31_stack[vm.qm31_sp++] = val;
                break;
            }

            case OP_DUP_QM31: {
                uint8_t depth = bytecode[vm.ip++];
                QM31 val = vm.qm31_stack[vm.qm31_sp - 1 - depth];
                vm.qm31_stack[vm.qm31_sp++] = val;
                break;
            }

            case OP_POP_QM31: {
                vm.qm31_sp--;
                break;
            }

            case OP_SWAP_QM31: {
                uint8_t depth1 = bytecode[vm.ip++];
                uint8_t depth2 = bytecode[vm.ip++];
                QM31 temp = vm.qm31_stack[vm.qm31_sp - 1 - depth1];
                vm.qm31_stack[vm.qm31_sp - 1 - depth1] = vm.qm31_stack[vm.qm31_sp - 1 - depth2];
                vm.qm31_stack[vm.qm31_sp - 1 - depth2] = temp;
                break;
            }

            case OP_LOAD_RANDOM_COEFF: {
                uint16_t index = read_u16_be(bytecode, &vm.ip);
                QM31 coeff = load_qm31(&random_coeffs[index * 4]);
                vm.qm31_stack[vm.qm31_sp++] = coeff;
                break;
            }

            // QM31 Arithmetic Operations
            case OP_ADD_QM31: {
                QM31 b = vm.qm31_stack[--vm.qm31_sp];
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_add(a, b);
                break;
            }

            case OP_SUB_QM31: {
                QM31 b = vm.qm31_stack[--vm.qm31_sp];
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_sub(a, b);
                break;
            }

            case OP_MUL_QM31: {
                QM31 b = vm.qm31_stack[--vm.qm31_sp];
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_mul(a, b);
                break;
            }

            case OP_NEG_QM31: {
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_neg(a);
                break;
            }

            case OP_SQUARE_QM31: {
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_square(a);
                break;
            }

            case OP_INV_QM31: {
                QM31 a = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_inv(a);
                break;
            }

            // Type Conversion Operations
            case OP_M31_TO_QM31: {
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                QM31 result;
                result.c0.a = a;
                result.c0.b = 0;
                result.c1.a = 0;
                result.c1.b = 0;
                vm.qm31_stack[vm.qm31_sp++] = result;
                break;
            }

            case OP_COMBINE_EF: {
                // Pop 4 M31 values in reverse order and combine into QM31
                uint32_t d = vm.m31_stack[--vm.m31_sp];
                uint32_t c = vm.m31_stack[--vm.m31_sp];
                uint32_t b = vm.m31_stack[--vm.m31_sp];
                uint32_t a = vm.m31_stack[--vm.m31_sp];
                QM31 result;
                result.c0.a = a;
                result.c0.b = b;
                result.c1.a = c;
                result.c1.b = d;
                vm.qm31_stack[vm.qm31_sp++] = result;
                break;
            }

            case OP_MUL_M31_QM31: {
                uint32_t scalar = vm.m31_stack[--vm.m31_sp];
                QM31 q = vm.qm31_stack[--vm.qm31_sp];
                vm.qm31_stack[vm.qm31_sp++] = qm31_mul_m31(q, scalar);
                break;
            }

            case OP_ADD_M31_QM31: {
                uint32_t scalar = vm.m31_stack[--vm.m31_sp];
                QM31 q = vm.qm31_stack[--vm.qm31_sp];
                QM31 scalar_qm31;
                scalar_qm31.c0.a = scalar;
                scalar_qm31.c0.b = 0;
                scalar_qm31.c1.a = 0;
                scalar_qm31.c1.b = 0;
                vm.qm31_stack[vm.qm31_sp++] = qm31_add(q, scalar_qm31);
                break;
            }

            // Constraint Operations
            case OP_ADD_CONSTRAINT: {
                QM31 constraint = vm.qm31_stack[--vm.qm31_sp];
                QM31 coeff = load_qm31(&random_coeffs[vm.constraint_idx * 4]);
                vm.constraint_accum = qm31_add(vm.constraint_accum,
                                               qm31_mul(constraint, coeff));
                vm.constraint_idx++;
                break;
            }

            case OP_MARK_INTERMEDIATE: {
                // Skip the is_extension byte, it's only for debugging
                vm.ip++;
                break;
            }

            // Control Flow Operations (simplified for now)
            case OP_JUMP_IF_ZERO: {
                // Skip offset bytes (not implemented yet)
                vm.ip += 2;
                // For now, we don't implement conditional jumps
                // This would require checking if top QM31 is zero
                break;
            }

            case OP_JUMP: {
                // Skip offset bytes (not implemented yet)
                vm.ip += 2;
                // For now, we don't implement unconditional jumps
                break;
            }

            // Logup Operations (placeholder for now)
            case OP_WRITE_LOGUP_FRAC: {
                // Pop denominator and numerator from QM31 stack
                vm.qm31_sp -= 2;
                // TODO: Implement logup accumulation
                break;
            }

            case OP_FINALIZE_LOGUP: {
                // TODO: Implement logup finalization
                break;
            }

            // Program End
            case OP_PROGRAM_END: {
                // Exit the interpreter loop
                vm.ip = bytecode_len;
                break;
            }

            default: {
                // Unknown opcode - exit
                vm.ip = bytecode_len;
                break;
            }
        }
    }

    // Finalize: multiply accumulator by denominator inverse
    // Note: denom_inv buffer is already expanded to full eval_domain_size
    QM31 denom_inv_val = load_qm31(&denom_inv[row_idx * 4]);
    QM31 result = qm31_mul(vm.constraint_accum, denom_inv_val);

    // Store result to output
    store_qm31(&output[row_idx * 4], result);
}

/// Trace reshape kernel - converts trace columns from row-major to column-major layout
/// This kernel reshapes a single column of the trace for efficient GPU access patterns.
///
/// Parameters:
/// - src_column: Input column data in row-major layout
/// - dst_flattened: Output buffer for column-major layout
/// - col_idx: Index of this column in the flattened output
/// - n_rows: Number of rows in the column
kernel void trace_reshape_column(
    device const uint32_t* src_column [[buffer(0)]],
    device uint32_t* dst_flattened [[buffer(1)]],
    constant uint32_t& col_idx [[buffer(2)]],
    constant uint32_t& n_rows [[buffer(3)]],
    uint row_idx [[thread_position_in_grid]]
) {
    // Bounds check
    if (row_idx >= n_rows) {
        return;
    }

    // Copy from row-major column to column-major flattened layout
    // src_column[row_idx] -> dst_flattened[col_idx * n_rows + row_idx]
    dst_flattened[col_idx * n_rows + row_idx] = src_column[row_idx];
}

/// Batched trace reshape kernel - reshapes multiple columns in parallel
/// This kernel reshapes multiple columns from row-major to column-major layout.
///
/// Parameters:
/// - src_flattened: Source columns in flattened array (columns are concatenated)
/// - dst_flattened: Output buffer for column-major layout
/// - n_columns: Number of columns to reshape
/// - n_rows: Number of rows per column
kernel void trace_reshape_batch(
    device const uint32_t* src_flattened [[buffer(0)]],
    device uint32_t* dst_flattened [[buffer(1)]],
    constant uint32_t& n_columns [[buffer(2)]],
    constant uint32_t& n_rows [[buffer(3)]],
    uint2 thread_idx [[thread_position_in_grid]]
) {
    uint col_idx = thread_idx.x;
    uint row_idx = thread_idx.y;

    // Bounds check
    if (col_idx >= n_columns || row_idx >= n_rows) {
        return;
    }

    // Read from source (columns are concatenated)
    uint src_offset = col_idx * n_rows + row_idx;
    uint32_t value = src_flattened[src_offset];

    // Write to column-major destination
    dst_flattened[col_idx * n_rows + row_idx] = value;
}
