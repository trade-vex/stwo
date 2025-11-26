#[cfg(test)]
mod tests {
    use crate::{TraceLocationAllocator, FrameworkComponent};
    use num_traits::Zero;
    use stwo::core::fields::qm31::SecureField;

    // Simple test component for GPU constraint evaluation
    #[derive(Clone)]
    struct TestEval {
        log_size: u32,
    }

    impl crate::FrameworkEval for TestEval {
        fn log_size(&self) -> u32 {
            self.log_size
        }

        fn max_constraint_log_degree_bound(&self) -> u32 {
            self.log_size + 1
        }

        fn evaluate<E: crate::EvalAtRow>(&self, mut eval: E) -> E {
            // Simple constraint: a[0] * a[1] = a[2]
            let a0 = eval.next_trace_mask();
            let a1 = eval.next_trace_mask();
            let a2 = eval.next_trace_mask();

            eval.add_constraint(a0 * a1 - a2);
            eval
        }
    }

    #[test]
    #[cfg(all(target_os = "macos", feature = "metal_prover"))]
    fn test_gpu_constraint_bytecode_generation() {
        println!("\n=== Testing GPU Constraint Bytecode Generation ===");

        // Test with log_n=12 (minimum threshold for GPU)
        let log_n = 12;
        println!("Creating component with log_n={} (2^{} = {} rows)", log_n, log_n, 1 << log_n);

        let component = FrameworkComponent::new(
            &mut TraceLocationAllocator::default(),
            TestEval { log_size: log_n },
            SecureField::zero(),
        );

        // Check if bytecode was generated
        if let Some(bytecode) = component.bytecode() {
            println!("✓ Bytecode generated: {} bytes", bytecode.len());

            // Verify bytecode has reasonable size
            assert!(bytecode.len() > 10, "Bytecode should not be empty");

            // Print first few opcodes for debugging
            println!("  First 16 bytes: {:02x?}", &bytecode[..bytecode.len().min(16)]);

            // Check for expected opcodes
            // OP_PROGRAM_END = 0xFF should be present
            assert!(bytecode.contains(&0xFF), "Bytecode should contain OP_PROGRAM_END");
        } else {
            panic!("No bytecode generated for GPU constraint evaluation!");
        }

        // Test with larger size
        let log_n = 16;
        println!("\nCreating component with log_n={} (2^{} = {} rows)", log_n, log_n, 1 << log_n);

        let component = FrameworkComponent::new(
            &mut TraceLocationAllocator::default(),
            TestEval { log_size: log_n },
            SecureField::zero(),
        );

        if let Some(bytecode) = component.bytecode() {
            println!("✓ Bytecode generated: {} bytes", bytecode.len());
            assert!(bytecode.len() > 10, "Bytecode should not be empty");
        } else {
            panic!("No bytecode generated for large component!");
        }

        println!("\n✅ GPU constraint bytecode generation test passed!");
    }

    #[test]
    #[cfg(not(all(target_os = "macos", feature = "metal_prover")))]
    fn test_gpu_constraint_bytecode_generation() {
        println!("Skipping GPU constraint test (not on macOS with metal_prover feature)");
    }
}