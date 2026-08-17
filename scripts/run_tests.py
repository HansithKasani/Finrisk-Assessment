"""
Test runner script for Credit Risk Assessment System
Provides convenient interface for running different test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd):
    """Run a command and return exit code"""
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print('='*70 + '\n')
    
    result = subprocess.run(cmd)
    return result.returncode


def run_all_tests(verbose=True, coverage=True):
    """Run all tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/']
    
    if verbose:
        cmd.append('-v')
    
    if coverage:
        cmd.extend(['--cov=src', '--cov-report=html', '--cov-report=term'])
    
    return run_command(cmd)


def run_unit_tests():
    """Run only unit tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-m', 'unit', '-v']
    return run_command(cmd)


def run_integration_tests():
    """Run only integration tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-m', 'integration', '-v']
    return run_command(cmd)


def run_specific_test(test_file):
    """Run a specific test file"""
    cmd = [sys.executable, '-m', 'pytest', f'tests/{test_file}', '-v']
    return run_command(cmd)


def run_quick_tests():
    """Run quick tests (exclude slow tests)"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '-m', 'not slow', '-v']
    return run_command(cmd)


def run_with_failures_only():
    """Run only previously failed tests"""
    cmd = [sys.executable, '-m', 'pytest', 'tests/', '--lf', '-v']
    return run_command(cmd)


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description='Run tests for Credit Risk Assessment System'
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'unit', 'integration', 'quick', 'failed'],
        default='all',
        help='Test mode to run'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Specific test file to run (e.g., test_preprocessor.py)'
    )
    
    parser.add_argument(
        '--no-coverage',
        action='store_true',
        help='Skip coverage reporting'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("AI Credit Risk Assessment - Test Suite")
    print("="*70)
    
    # Run specific file if provided
    if args.file:
        print(f"\nRunning specific test file: {args.file}")
        exit_code = run_specific_test(args.file)
    
    # Run based on mode
    elif args.mode == 'all':
        print("\nRunning all tests...")
        exit_code = run_all_tests(
            verbose=not args.quiet,
            coverage=not args.no_coverage
        )
    
    elif args.mode == 'unit':
        print("\nRunning unit tests only...")
        exit_code = run_unit_tests()
    
    elif args.mode == 'integration':
        print("\nRunning integration tests only...")
        exit_code = run_integration_tests()
    
    elif args.mode == 'quick':
        print("\nRunning quick tests (excluding slow tests)...")
        exit_code = run_quick_tests()
    
    elif args.mode == 'failed':
        print("\nRunning previously failed tests...")
        exit_code = run_with_failures_only()
    
    else:
        print(f"\nUnknown mode: {args.mode}")
        exit_code = 1
    
    # Print summary
    print("\n" + "="*70)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*70 + "\n")
    
    # Show coverage report location
    if args.mode == 'all' and not args.no_coverage:
        coverage_dir = Path(__file__).parent.parent / "htmlcov"
        if coverage_dir.exists():
            print(f"📊 Coverage report: {coverage_dir / 'index.html'}")
            print(f"   Open in browser to view detailed coverage\n")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
