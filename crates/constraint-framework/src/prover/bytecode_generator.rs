/// Bytecode generator for constraint evaluation.
///
/// This module implements a "recording evaluator" that captures constraint operations
/// as bytecode instead of executing them. The bytecode can then be interpreted on the GPU.

use std::marker::PhantomData;

use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use stwo::core::Fraction;

use super::bytecode::{BytecodeProgram, Instruction};
use crate::{EvalAtRow, INTERACTION_TRACE_IDX, Batching};
use crate::logup::LogupAtRow;

/// Expression tree node for M31 operations.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum M31Expr {
    /// Load from trace column.
    LoadTrace { interaction: u8, col_idx: u16, offset: i16 },
    /// Constant value.
    Const(u32),
    /// Addition.
    Add(Box<M31Expr>, Box<M31Expr>),
    /// Subtraction.
    Sub(Box<M31Expr>, Box<M31Expr>),
    /// Multiplication.
    Mul(Box<M31Expr>, Box<M31Expr>),
    /// Negation.
    Neg(Box<M31Expr>),
    /// Square.
    Square(Box<M31Expr>),
    /// Inverse.
    Inv(Box<M31Expr>),
}

/// Expression tree node for QM31 operations.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub enum QM31Expr {
    /// Load from trace column (4 consecutive M31 columns).
    LoadTrace { interaction: u8, col_idx: u16, offset: i16 },
    /// Constant value.
    Const([u32; 4]),
    /// Random coefficient.
    RandomCoeff(u16),
    /// Addition.
    Add(Box<QM31Expr>, Box<QM31Expr>),
    /// Subtraction.
    Sub(Box<QM31Expr>, Box<QM31Expr>),
    /// Multiplication.
    Mul(Box<QM31Expr>, Box<QM31Expr>),
    /// Negation.
    Neg(Box<QM31Expr>),
    /// Square.
    Square(Box<QM31Expr>),
    /// Inverse.
    Inv(Box<QM31Expr>),
    /// Convert M31 to QM31.
    FromM31(Box<M31Expr>),
    /// Combine 4 M31 into QM31.
    CombineEF(Box<M31Expr>, Box<M31Expr>, Box<M31Expr>, Box<M31Expr>),
    /// Multiply QM31 by M31.
    MulM31(Box<QM31Expr>, Box<M31Expr>),
    /// Add QM31 and M31.
    AddM31(Box<QM31Expr>, Box<M31Expr>),
    /// Multiply QM31 by SecureField constant.
    MulSecureField(Box<QM31Expr>, [u32; 4]),
    /// Add QM31 and SecureField constant.
    AddSecureField(Box<QM31Expr>, [u32; 4]),
}

/// Expression handle for M31 (wrapper around expression tree).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct M31Handle {
    expr: M31Expr,
}

/// Expression handle for QM31 (wrapper around expression tree).
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct QM31Handle {
    expr: QM31Expr,
}

/// Recording evaluator that generates bytecode.
#[allow(dead_code)]
pub struct BytecodeGenerator {
    /// Bytecode program being built.
    pub program: BytecodeProgram,

    /// Column index per interaction (tracks which column to read next).
    column_index_per_interaction: Vec<usize>,

    /// Logup state (placeholder for now - not yet implemented).
    #[allow(dead_code)]
    pub logup: LogupAtRow<Self>,

    /// List of constraint expressions to compile.
    constraints: Vec<QM31Expr>,

    /// Phantom data for FrameworkEval type parameter.
    _phantom: PhantomData<()>,
}

#[allow(dead_code)]
impl BytecodeGenerator {
    pub fn new(n_interactions: usize, log_size: u32, claimed_sum: SecureField) -> Self {
        Self {
            program: BytecodeProgram::new(),
            column_index_per_interaction: vec![0; n_interactions],
            logup: LogupAtRow::new(INTERACTION_TRACE_IDX, claimed_sum, log_size),
            constraints: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Compile all constraints to bytecode.
    pub fn compile(&mut self) {
        // Track current stack depths during compilation.
        let mut m31_depth = 0usize;
        let mut qm31_depth = 0usize;

        // Clone constraints to avoid borrow checker issues.
        let constraints = self.constraints.clone();

        for constraint in &constraints {
            // Compile each constraint expression to bytecode.
            // This will leave the result on the QM31 stack.
            self.compile_qm31_expr(constraint, &mut m31_depth, &mut qm31_depth);

            // Add constraint instruction (pops QM31 from stack).
            self.program.encode(Instruction::AddConstraint);
            qm31_depth -= 1;
        }

        // Emit ProgramEnd marker for batched evaluation.
        // This tells the batched VM to finalize results for this component
        // and move to the next component's bytecode.
        self.program.encode(Instruction::ProgramEnd);
    }

    /// Compile M31 expression to bytecode (leaves result on M31 stack).
    fn compile_m31_expr(&mut self, expr: &M31Expr, m31_depth: &mut usize, qm31_depth: &mut usize) {
        match expr {
            M31Expr::LoadTrace { interaction, col_idx, offset } => {
                self.program.encode(Instruction::LoadTraceM31 {
                    interaction: *interaction,
                    col_idx: *col_idx,
                    offset: *offset,
                });
                *m31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            M31Expr::Const(value) => {
                self.program.encode(Instruction::LoadConstM31 { value: *value });
                *m31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            M31Expr::Add(lhs, rhs) => {
                self.compile_m31_expr(lhs, m31_depth, qm31_depth);
                self.compile_m31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::AddM31);
                *m31_depth -= 1;  // Pops 2, pushes 1
            }
            M31Expr::Sub(lhs, rhs) => {
                self.compile_m31_expr(lhs, m31_depth, qm31_depth);
                self.compile_m31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::SubM31);
                *m31_depth -= 1;
            }
            M31Expr::Mul(lhs, rhs) => {
                self.compile_m31_expr(lhs, m31_depth, qm31_depth);
                self.compile_m31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::MulM31);
                *m31_depth -= 1;
            }
            M31Expr::Neg(inner) => {
                self.compile_m31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::NegM31);
                // Depth unchanged (pops 1, pushes 1)
            }
            M31Expr::Square(inner) => {
                self.compile_m31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::SquareM31);
            }
            M31Expr::Inv(inner) => {
                self.compile_m31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::InvM31);
            }
        }
    }

    /// Compile QM31 expression to bytecode (leaves result on QM31 stack).
    fn compile_qm31_expr(&mut self, expr: &QM31Expr, m31_depth: &mut usize, qm31_depth: &mut usize) {
        match expr {
            QM31Expr::LoadTrace { interaction, col_idx, offset } => {
                self.program.encode(Instruction::LoadTraceQM31 {
                    interaction: *interaction,
                    col_idx: *col_idx,
                    offset: *offset,
                });
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            QM31Expr::Const(values) => {
                self.program.encode(Instruction::LoadConstQM31 { values: *values });
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            QM31Expr::RandomCoeff(index) => {
                self.program.encode(Instruction::LoadRandomCoeff { index: *index });
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            QM31Expr::Add(lhs, rhs) => {
                self.compile_qm31_expr(lhs, m31_depth, qm31_depth);
                self.compile_qm31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::AddQM31);
                *qm31_depth -= 1;
            }
            QM31Expr::Sub(lhs, rhs) => {
                self.compile_qm31_expr(lhs, m31_depth, qm31_depth);
                self.compile_qm31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::SubQM31);
                *qm31_depth -= 1;
            }
            QM31Expr::Mul(lhs, rhs) => {
                self.compile_qm31_expr(lhs, m31_depth, qm31_depth);
                self.compile_qm31_expr(rhs, m31_depth, qm31_depth);
                self.program.encode(Instruction::MulQM31);
                *qm31_depth -= 1;
            }
            QM31Expr::Neg(inner) => {
                self.compile_qm31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::NegQM31);
            }
            QM31Expr::Square(inner) => {
                self.compile_qm31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::SquareQM31);
            }
            QM31Expr::Inv(inner) => {
                self.compile_qm31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::InvQM31);
            }
            QM31Expr::FromM31(inner) => {
                self.compile_m31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::M31ToQM31);
                *m31_depth -= 1;
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            QM31Expr::CombineEF(e0, e1, e2, e3) => {
                self.compile_m31_expr(e0, m31_depth, qm31_depth);
                self.compile_m31_expr(e1, m31_depth, qm31_depth);
                self.compile_m31_expr(e2, m31_depth, qm31_depth);
                self.compile_m31_expr(e3, m31_depth, qm31_depth);
                self.program.encode(Instruction::CombineEF);
                *m31_depth -= 4;
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
            }
            QM31Expr::MulM31(qm31_expr, m31_expr) => {
                self.compile_m31_expr(m31_expr, m31_depth, qm31_depth);
                self.compile_qm31_expr(qm31_expr, m31_depth, qm31_depth);
                self.program.encode(Instruction::MulM31QM31);
                *m31_depth -= 1;
                // QM31 depth unchanged (pops 1, pushes 1)
            }
            QM31Expr::AddM31(qm31_expr, m31_expr) => {
                self.compile_m31_expr(m31_expr, m31_depth, qm31_depth);
                self.compile_qm31_expr(qm31_expr, m31_depth, qm31_depth);
                self.program.encode(Instruction::AddM31QM31);
                *m31_depth -= 1;
            }
            QM31Expr::MulSecureField(inner, values) => {
                self.compile_qm31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::LoadConstQM31 { values: *values });
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
                self.program.encode(Instruction::MulQM31);
                *qm31_depth -= 1;
            }
            QM31Expr::AddSecureField(inner, values) => {
                self.compile_qm31_expr(inner, m31_depth, qm31_depth);
                self.program.encode(Instruction::LoadConstQM31 { values: *values });
                *qm31_depth += 1;
                self.program.update_stack_depth(*m31_depth, *qm31_depth);
                self.program.encode(Instruction::AddQM31);
                *qm31_depth -= 1;
            }
        }
    }
}

impl EvalAtRow for BytecodeGenerator {
    type F = M31Handle;
    type EF = QM31Handle;

    fn next_interaction_mask<const N: usize>(
        &mut self,
        interaction: usize,
        offsets: [isize; N],
    ) -> [Self::F; N] {
        let col_idx = self.column_index_per_interaction[interaction];
        self.column_index_per_interaction[interaction] += 1;

        offsets.map(|offset| {
            M31Handle {
                expr: M31Expr::LoadTrace {
                    interaction: interaction as u8,
                    col_idx: col_idx as u16,
                    offset: offset as i16,
                },
            }
        })
    }

    fn add_constraint<G>(&mut self, constraint: G)
    where
        Self::EF: std::ops::Mul<G, Output = Self::EF> + From<G>,
    {
        // Convert G to QM31Handle and add to constraints list.
        let qm31_constraint = QM31Handle::from(constraint);
        self.constraints.push(qm31_constraint.expr);
    }

    fn combine_ef(values: [Self::F; 4]) -> Self::EF {
        QM31Handle {
            expr: QM31Expr::CombineEF(
                Box::new(values[0].expr.clone()),
                Box::new(values[1].expr.clone()),
                Box::new(values[2].expr.clone()),
                Box::new(values[3].expr.clone()),
            ),
        }
    }

    fn write_logup_frac(&mut self, _fraction: Fraction<Self::EF, Self::EF>) {
        // TODO: Implement logup bytecode generation.
        // For now, this is a no-op.
    }

    fn finalize_logup_batched(&mut self, _batching: &Batching) {
        // TODO: Implement logup finalization bytecode.
    }

    fn finalize_logup(&mut self) {
        // TODO: Implement logup finalization bytecode.
    }

    fn finalize_logup_in_pairs(&mut self) {
        // TODO: Implement logup finalization bytecode.
    }
}

// ===== Arithmetic operations on M31Handle (build expression trees) =====

impl std::ops::Add for M31Handle {
    type Output = M31Handle;
    fn add(self, rhs: M31Handle) -> M31Handle {
        M31Handle {
            expr: M31Expr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Sub for M31Handle {
    type Output = M31Handle;
    fn sub(self, rhs: M31Handle) -> M31Handle {
        M31Handle {
            expr: M31Expr::Sub(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Mul for M31Handle {
    type Output = M31Handle;
    fn mul(self, rhs: M31Handle) -> M31Handle {
        M31Handle {
            expr: M31Expr::Mul(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Neg for M31Handle {
    type Output = M31Handle;
    fn neg(self) -> M31Handle {
        M31Handle {
            expr: M31Expr::Neg(Box::new(self.expr)),
        }
    }
}

// ===== Arithmetic operations on QM31Handle (build expression trees) =====

impl std::ops::Add for QM31Handle {
    type Output = QM31Handle;
    fn add(self, rhs: QM31Handle) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Sub for QM31Handle {
    type Output = QM31Handle;
    fn sub(self, rhs: QM31Handle) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::Sub(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Mul for QM31Handle {
    type Output = QM31Handle;
    fn mul(self, rhs: QM31Handle) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::Mul(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Neg for QM31Handle {
    type Output = QM31Handle;
    fn neg(self) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::Neg(Box::new(self.expr)),
        }
    }
}

// ===== Zero trait (required by EvalAtRow) =====

impl num_traits::Zero for M31Handle {
    fn zero() -> Self {
        M31Handle {
            expr: M31Expr::Const(0),
        }
    }

    fn is_zero(&self) -> bool {
        false  // We don't track actual values, only expression trees
    }
}

impl num_traits::Zero for QM31Handle {
    fn zero() -> Self {
        QM31Handle {
            expr: QM31Expr::Const([0, 0, 0, 0]),
        }
    }

    fn is_zero(&self) -> bool {
        false
    }
}

impl num_traits::One for M31Handle {
    fn one() -> Self {
        M31Handle {
            expr: M31Expr::Const(1),
        }
    }
}

impl num_traits::One for QM31Handle {
    fn one() -> Self {
        QM31Handle {
            expr: QM31Expr::Const([1, 0, 0, 0]),
        }
    }
}

// ===== From conversions =====

impl From<BaseField> for M31Handle {
    fn from(value: BaseField) -> Self {
        M31Handle {
            expr: M31Expr::Const(value.0),
        }
    }
}

impl From<SecureField> for QM31Handle {
    fn from(value: SecureField) -> Self {
        let m31_array = value.to_m31_array();
        QM31Handle {
            expr: QM31Expr::Const([m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0]),
        }
    }
}

impl From<M31Handle> for QM31Handle {
    fn from(value: M31Handle) -> Self {
        QM31Handle {
            expr: QM31Expr::FromM31(Box::new(value.expr)),
        }
    }
}

// ===== AddAssign trait =====

impl std::ops::AddAssign for M31Handle {
    fn add_assign(&mut self, rhs: M31Handle) {
        *self = M31Handle {
            expr: M31Expr::Add(Box::new(self.expr.clone()), Box::new(rhs.expr)),
        };
    }
}

impl std::ops::AddAssign for QM31Handle {
    fn add_assign(&mut self, rhs: QM31Handle) {
        *self = QM31Handle {
            expr: QM31Expr::Add(Box::new(self.expr.clone()), Box::new(rhs.expr)),
        };
    }
}

impl std::ops::AddAssign<BaseField> for M31Handle {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = M31Handle {
            expr: M31Expr::Add(Box::new(self.expr.clone()), Box::new(M31Expr::Const(rhs.0))),
        };
    }
}

impl std::ops::AddAssign<BaseField> for QM31Handle {
    fn add_assign(&mut self, rhs: BaseField) {
        *self = QM31Handle {
            expr: QM31Expr::AddSecureField(Box::new(self.expr.clone()), [rhs.0, 0, 0, 0]),
        };
    }
}

// ===== Mul trait with BaseField =====

impl std::ops::Mul<BaseField> for M31Handle {
    type Output = M31Handle;
    fn mul(self, rhs: BaseField) -> M31Handle {
        M31Handle {
            expr: M31Expr::Mul(Box::new(self.expr), Box::new(M31Expr::Const(rhs.0))),
        }
    }
}

impl std::ops::Mul<BaseField> for QM31Handle {
    type Output = QM31Handle;
    fn mul(self, rhs: BaseField) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::MulSecureField(Box::new(self.expr), [rhs.0, 0, 0, 0]),
        }
    }
}

// ===== Add/Sub/Mul with SecureField =====

impl std::ops::Add<SecureField> for M31Handle {
    type Output = QM31Handle;
    fn add(self, rhs: SecureField) -> QM31Handle {
        let m31_array = rhs.to_m31_array();
        QM31Handle {
            expr: QM31Expr::AddM31(
                Box::new(QM31Expr::Const([m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0])),
                Box::new(self.expr),
            ),
        }
    }
}

impl std::ops::Add<SecureField> for QM31Handle {
    type Output = QM31Handle;
    fn add(self, rhs: SecureField) -> QM31Handle {
        let m31_array = rhs.to_m31_array();
        QM31Handle {
            expr: QM31Expr::AddSecureField(
                Box::new(self.expr),
                [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0],
            ),
        }
    }
}

impl std::ops::Sub<SecureField> for QM31Handle {
    type Output = QM31Handle;
    fn sub(self, rhs: SecureField) -> QM31Handle {
        let m31_array = rhs.to_m31_array();
        let neg_rhs = [
            0u32.wrapping_sub(m31_array[0].0) & 0x7FFFFFFF,
            0u32.wrapping_sub(m31_array[1].0) & 0x7FFFFFFF,
            0u32.wrapping_sub(m31_array[2].0) & 0x7FFFFFFF,
            0u32.wrapping_sub(m31_array[3].0) & 0x7FFFFFFF,
        ];
        QM31Handle {
            expr: QM31Expr::AddSecureField(Box::new(self.expr), neg_rhs),
        }
    }
}

impl std::ops::Mul<SecureField> for M31Handle {
    type Output = QM31Handle;
    fn mul(self, rhs: SecureField) -> QM31Handle {
        let m31_array = rhs.to_m31_array();
        QM31Handle {
            expr: QM31Expr::MulM31(
                Box::new(QM31Expr::Const([m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0])),
                Box::new(self.expr),
            ),
        }
    }
}

impl std::ops::Mul<SecureField> for QM31Handle {
    type Output = QM31Handle;
    fn mul(self, rhs: SecureField) -> QM31Handle {
        let m31_array = rhs.to_m31_array();
        QM31Handle {
            expr: QM31Expr::MulSecureField(
                Box::new(self.expr),
                [m31_array[0].0, m31_array[1].0, m31_array[2].0, m31_array[3].0],
            ),
        }
    }
}

// ===== Cross-type operations (M31 + QM31) =====

impl std::ops::Add<M31Handle> for QM31Handle {
    type Output = QM31Handle;
    fn add(self, rhs: M31Handle) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::AddM31(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Mul<M31Handle> for QM31Handle {
    type Output = QM31Handle;
    fn mul(self, rhs: M31Handle) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::MulM31(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl std::ops::Add<BaseField> for QM31Handle {
    type Output = QM31Handle;
    fn add(self, rhs: BaseField) -> QM31Handle {
        QM31Handle {
            expr: QM31Expr::AddSecureField(Box::new(self.expr), [rhs.0, 0, 0, 0]),
        }
    }
}

// ===== MulAssign trait (required by FieldExpOps) =====

impl std::ops::MulAssign for M31Handle {
    fn mul_assign(&mut self, rhs: M31Handle) {
        *self = M31Handle {
            expr: M31Expr::Mul(Box::new(self.expr.clone()), Box::new(rhs.expr)),
        };
    }
}

impl std::ops::MulAssign for QM31Handle {
    fn mul_assign(&mut self, rhs: QM31Handle) {
        *self = QM31Handle {
            expr: QM31Expr::Mul(Box::new(self.expr.clone()), Box::new(rhs.expr)),
        };
    }
}

// ===== FieldExpOps trait (for square() method used in constraints) =====

impl stwo::core::fields::FieldExpOps for M31Handle {
    fn square(&self) -> Self {
        M31Handle {
            expr: M31Expr::Square(Box::new(self.expr.clone())),
        }
    }

    fn inverse(&self) -> Self {
        M31Handle {
            expr: M31Expr::Inv(Box::new(self.expr.clone())),
        }
    }
}

impl stwo::core::fields::FieldExpOps for QM31Handle {
    fn square(&self) -> Self {
        QM31Handle {
            expr: QM31Expr::Square(Box::new(self.expr.clone())),
        }
    }

    fn inverse(&self) -> Self {
        QM31Handle {
            expr: QM31Expr::Inv(Box::new(self.expr.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_bytecode_generator_basic() {
        // TODO: Add tests for bytecode generator
        // The bytecode generator is tested through integration tests
    }
}
