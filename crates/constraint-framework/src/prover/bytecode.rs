/// Bytecode instruction set for GPU constraint evaluation.
///
/// This module defines a stack-based bytecode ISA for expressing constraint evaluation
/// operations. The bytecode is generated on the CPU during AIR compilation and then
/// interpreted on the GPU by the Metal VM kernel.
///
/// ## Stack Machine Architecture
///
/// The VM uses two stacks:
/// - **M31 stack**: For BaseField (M31) values
/// - **QM31 stack**: For SecureField (QM31) values
///
/// Instructions pop operands from the stack and push results back.
///
/// ## Bytecode Format
///
/// Each instruction is encoded as:
/// - 1 byte: opcode
/// - 0-8 bytes: immediate operands (depending on opcode)
///
/// ## Example Bytecode Sequence
///
/// For the Fibonacci constraint: `c - (a^2 + b^2) = 0`
///
/// ```text
/// LOAD_TRACE_M31  0, 0, 0    // Load column 0, offset 0 -> stack: [a]
/// LOAD_TRACE_M31  0, 1, 0    // Load column 1, offset 0 -> stack: [a, b]
/// LOAD_TRACE_M31  0, 2, 0    // Load column 2, offset 0 -> stack: [a, b, c]
/// SWAP_M31        0, 2       // Swap top and 3rd elem    -> stack: [c, b, a]
/// DUP_M31         0          // Duplicate top            -> stack: [c, b, a, a]
/// SQUARE_M31                 // Square top               -> stack: [c, b, a, a^2]
/// SWAP_M31        0, 2       // Bring b to top           -> stack: [c, a^2, a, b]
/// DUP_M31         0          // Duplicate b              -> stack: [c, a^2, a, b, b]
/// SQUARE_M31                 // Square b                 -> stack: [c, a^2, a, b^2]
/// ADD_M31                    // a^2 + b^2                -> stack: [c, a^2, a+b^2]
/// SUB_M31                    // c - (a^2 + b^2)          -> stack: [result]
/// M31_TO_QM31                // Convert to QM31          -> qm31_stack: [result_qm31]
/// ADD_CONSTRAINT             // Submit constraint
/// ```

use std_shims::{vec, Vec};

/// Bytecode instruction opcodes.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    // === M31 Stack Operations ===
    /// Load M31 value from trace column.
    /// Args: interaction(u8), col_idx(u16), offset(i16)
    LoadTraceM31 = 0x00,

    /// Load M31 constant.
    /// Args: value(u32)
    LoadConstM31 = 0x01,

    /// Duplicate M31 stack value at depth.
    /// Args: depth(u8) - 0 = top of stack
    DupM31 = 0x02,

    /// Pop M31 stack top.
    PopM31 = 0x03,

    /// Swap two M31 stack elements.
    /// Args: depth1(u8), depth2(u8)
    SwapM31 = 0x04,

    // === M31 Arithmetic ===
    /// Add: pop two M31, push sum.
    AddM31 = 0x10,

    /// Subtract: pop b, pop a, push a - b.
    SubM31 = 0x11,

    /// Multiply: pop two M31, push product.
    MulM31 = 0x12,

    /// Negate: pop M31, push -x.
    NegM31 = 0x13,

    /// Square: pop M31, push x^2.
    SquareM31 = 0x14,

    /// Inverse: pop M31, push 1/x.
    InvM31 = 0x15,

    // === QM31 Stack Operations ===
    /// Load QM31 value from trace column.
    /// Args: interaction(u8), col_idx(u16), offset(i16)
    LoadTraceQM31 = 0x20,

    /// Load QM31 constant.
    /// Args: value(4xu32)
    LoadConstQM31 = 0x21,

    /// Duplicate QM31 stack value at depth.
    /// Args: depth(u8)
    DupQM31 = 0x22,

    /// Pop QM31 stack top.
    PopQM31 = 0x23,

    /// Swap two QM31 stack elements.
    /// Args: depth1(u8), depth2(u8)
    SwapQM31 = 0x24,

    /// Load random coefficient.
    /// Args: index(u16)
    LoadRandomCoeff = 0x25,

    // === QM31 Arithmetic ===
    /// Add: pop two QM31, push sum.
    AddQM31 = 0x30,

    /// Subtract: pop b, pop a, push a - b.
    SubQM31 = 0x31,

    /// Multiply: pop two QM31, push product.
    MulQM31 = 0x32,

    /// Negate: pop QM31, push -x.
    NegQM31 = 0x33,

    /// Square: pop QM31, push x^2.
    SquareQM31 = 0x34,

    /// Inverse: pop QM31, push 1/x.
    InvQM31 = 0x35,

    // === Type Conversions ===
    /// Convert M31 to QM31.
    /// Pop M31, push as QM31.
    M31ToQM31 = 0x40,

    /// Combine 4 M31 values into QM31.
    /// Pop 4 M31 (in reverse order), push QM31.
    CombineEF = 0x41,

    /// Multiply M31 by QM31.
    /// Pop M31, pop QM31, push QM31.
    MulM31QM31 = 0x42,

    /// Add M31 to QM31.
    /// Pop M31, pop QM31, push QM31.
    AddM31QM31 = 0x43,

    // === Constraint Operations ===
    /// Add constraint to accumulator.
    /// Pop QM31 from stack, multiply by random coeff, add to accumulator.
    AddConstraint = 0x50,

    /// Mark intermediate value (no-op for now, useful for debugging).
    /// Args: is_extension(bool)
    MarkIntermediate = 0x51,

    // === Control Flow (future optimization) ===
    /// Jump to offset if top QM31 is zero (for conditional constraints).
    /// Args: offset(i16)
    JumpIfZero = 0x60,

    /// Unconditional jump.
    /// Args: offset(i16)
    Jump = 0x61,

    // === Logup Operations (for lookup arguments) ===
    /// Push logup fraction.
    /// Pop denominator QM31, pop numerator QM31.
    WriteLogupFrac = 0x70,

    /// Finalize logup (processes accumulated fractions).
    FinalizeLogup = 0x71,

    // === Program Control ===
    /// End of program marker (for batched evaluation).
    /// Signals the VM to finalize the current program's constraints,
    /// write results, reset state, and move to the next program.
    ProgramEnd = 0xFF,
}

/// Bytecode instruction with decoded operands.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // M31 Stack Ops
    LoadTraceM31 { interaction: u8, col_idx: u16, offset: i16 },
    LoadConstM31 { value: u32 },
    DupM31 { depth: u8 },
    PopM31,
    SwapM31 { depth1: u8, depth2: u8 },

    // M31 Arithmetic
    AddM31,
    SubM31,
    MulM31,
    NegM31,
    SquareM31,
    InvM31,

    // QM31 Stack Ops
    LoadTraceQM31 { interaction: u8, col_idx: u16, offset: i16 },
    LoadConstQM31 { values: [u32; 4] },
    DupQM31 { depth: u8 },
    PopQM31,
    SwapQM31 { depth1: u8, depth2: u8 },
    LoadRandomCoeff { index: u16 },

    // QM31 Arithmetic
    AddQM31,
    SubQM31,
    MulQM31,
    NegQM31,
    SquareQM31,
    InvQM31,

    // Type Conversions
    M31ToQM31,
    CombineEF,
    MulM31QM31,
    AddM31QM31,

    // Constraint Ops
    AddConstraint,
    MarkIntermediate { is_extension: bool },

    // Control Flow
    JumpIfZero { offset: i16 },
    Jump { offset: i16 },

    // Logup
    WriteLogupFrac,
    FinalizeLogup,

    // Program Control
    ProgramEnd,
}

/// Bytecode program (sequence of instructions).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    /// Raw bytecode bytes.
    pub bytes: Vec<u8>,
    /// Number of constraints in the program.
    pub n_constraints: usize,
    /// Maximum M31 stack depth required.
    pub max_m31_stack_depth: usize,
    /// Maximum QM31 stack depth required.
    pub max_qm31_stack_depth: usize,
}

#[allow(dead_code)]
impl BytecodeProgram {
    pub fn new() -> Self {
        Self {
            bytes: vec![],
            n_constraints: 0,
            max_m31_stack_depth: 0,
            max_qm31_stack_depth: 0,
        }
    }

    /// Encode an instruction to bytecode.
    pub fn encode(&mut self, instr: Instruction) {
        match instr {
            Instruction::LoadTraceM31 { interaction, col_idx, offset } => {
                self.bytes.push(Opcode::LoadTraceM31 as u8);
                self.bytes.push(interaction);
                self.bytes.extend_from_slice(&col_idx.to_be_bytes());
                self.bytes.extend_from_slice(&offset.to_be_bytes());
            }
            Instruction::LoadConstM31 { value } => {
                self.bytes.push(Opcode::LoadConstM31 as u8);
                self.bytes.extend_from_slice(&value.to_be_bytes());
            }
            Instruction::DupM31 { depth } => {
                self.bytes.push(Opcode::DupM31 as u8);
                self.bytes.push(depth);
            }
            Instruction::PopM31 => {
                self.bytes.push(Opcode::PopM31 as u8);
            }
            Instruction::SwapM31 { depth1, depth2 } => {
                self.bytes.push(Opcode::SwapM31 as u8);
                self.bytes.push(depth1);
                self.bytes.push(depth2);
            }

            // M31 arithmetic (no operands)
            Instruction::AddM31 => self.bytes.push(Opcode::AddM31 as u8),
            Instruction::SubM31 => self.bytes.push(Opcode::SubM31 as u8),
            Instruction::MulM31 => self.bytes.push(Opcode::MulM31 as u8),
            Instruction::NegM31 => self.bytes.push(Opcode::NegM31 as u8),
            Instruction::SquareM31 => self.bytes.push(Opcode::SquareM31 as u8),
            Instruction::InvM31 => self.bytes.push(Opcode::InvM31 as u8),

            Instruction::LoadTraceQM31 { interaction, col_idx, offset } => {
                self.bytes.push(Opcode::LoadTraceQM31 as u8);
                self.bytes.push(interaction);
                self.bytes.extend_from_slice(&col_idx.to_be_bytes());
                self.bytes.extend_from_slice(&offset.to_be_bytes());
            }
            Instruction::LoadConstQM31 { values } => {
                self.bytes.push(Opcode::LoadConstQM31 as u8);
                for val in &values {
                    self.bytes.extend_from_slice(&val.to_be_bytes());
                }
            }
            Instruction::DupQM31 { depth } => {
                self.bytes.push(Opcode::DupQM31 as u8);
                self.bytes.push(depth);
            }
            Instruction::PopQM31 => {
                self.bytes.push(Opcode::PopQM31 as u8);
            }
            Instruction::SwapQM31 { depth1, depth2 } => {
                self.bytes.push(Opcode::SwapQM31 as u8);
                self.bytes.push(depth1);
                self.bytes.push(depth2);
            }
            Instruction::LoadRandomCoeff { index } => {
                self.bytes.push(Opcode::LoadRandomCoeff as u8);
                self.bytes.extend_from_slice(&index.to_be_bytes());
            }

            // QM31 arithmetic (no operands)
            Instruction::AddQM31 => self.bytes.push(Opcode::AddQM31 as u8),
            Instruction::SubQM31 => self.bytes.push(Opcode::SubQM31 as u8),
            Instruction::MulQM31 => self.bytes.push(Opcode::MulQM31 as u8),
            Instruction::NegQM31 => self.bytes.push(Opcode::NegQM31 as u8),
            Instruction::SquareQM31 => self.bytes.push(Opcode::SquareQM31 as u8),
            Instruction::InvQM31 => self.bytes.push(Opcode::InvQM31 as u8),

            // Type conversions
            Instruction::M31ToQM31 => self.bytes.push(Opcode::M31ToQM31 as u8),
            Instruction::CombineEF => self.bytes.push(Opcode::CombineEF as u8),
            Instruction::MulM31QM31 => self.bytes.push(Opcode::MulM31QM31 as u8),
            Instruction::AddM31QM31 => self.bytes.push(Opcode::AddM31QM31 as u8),

            Instruction::AddConstraint => {
                self.bytes.push(Opcode::AddConstraint as u8);
                self.n_constraints += 1;
            }
            Instruction::MarkIntermediate { is_extension } => {
                self.bytes.push(Opcode::MarkIntermediate as u8);
                self.bytes.push(is_extension as u8);
            }

            Instruction::JumpIfZero { offset } => {
                self.bytes.push(Opcode::JumpIfZero as u8);
                self.bytes.extend_from_slice(&offset.to_be_bytes());
            }
            Instruction::Jump { offset } => {
                self.bytes.push(Opcode::Jump as u8);
                self.bytes.extend_from_slice(&offset.to_be_bytes());
            }

            Instruction::WriteLogupFrac => self.bytes.push(Opcode::WriteLogupFrac as u8),
            Instruction::FinalizeLogup => self.bytes.push(Opcode::FinalizeLogup as u8),

            Instruction::ProgramEnd => self.bytes.push(Opcode::ProgramEnd as u8),
        }
    }

    /// Update stack depth requirements (call after encoding each instruction).
    pub fn update_stack_depth(&mut self, m31_depth: usize, qm31_depth: usize) {
        self.max_m31_stack_depth = self.max_m31_stack_depth.max(m31_depth);
        self.max_qm31_stack_depth = self.max_qm31_stack_depth.max(qm31_depth);
    }
}

impl Default for BytecodeProgram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytecode_encoding() {
        let mut program = BytecodeProgram::new();

        // Encode: load M31 from column 0, offset 0
        program.encode(Instruction::LoadTraceM31 {
            interaction: 1,
            col_idx: 0,
            offset: 0,
        });

        // Encode: square
        program.encode(Instruction::SquareM31);

        // Encode: convert to QM31
        program.encode(Instruction::M31ToQM31);

        // Encode: add constraint
        program.encode(Instruction::AddConstraint);

        assert_eq!(program.n_constraints, 1);
        assert!(program.bytes.len() > 0);
    }
}
