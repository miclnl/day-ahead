#!/usr/bin/env python3
"""
Test Runner voor DAO Project
Voert alle tests uit en genereert een rapport
"""

import os
import sys
import subprocess
import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TestRunner:
    """Test runner voor alle DAO tests"""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.test_results = []
        self.start_time = datetime.datetime.now()

    def run_test_file(self, test_file: str, description: str) -> dict:
        """Run a single test file and return results"""
        logging.info(f"🧪 Running {description}...")

        try:
            # Change to project root directory
            os.chdir(self.project_root)

            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            success = result.returncode == 0

            # Parse output for test results
            output_lines = result.stdout.split('\n')
            test_count = 0
            passed_count = 0

            for line in output_lines:
                if '✅ PASS' in line:
                    passed_count += 1
                    test_count += 1
                elif '❌ FAIL' in line:
                    test_count += 1
                elif 'Test Results:' in line:
                    # Extract test summary
                    try:
                        parts = line.split(':')
                        if len(parts) >= 2:
                            summary = parts[1].strip()
                            if '/' in summary:
                                passed, total = summary.split('/')
                                passed_count = int(passed)
                                test_count = int(total.split()[0])
                    except:
                        pass

            result_info = {
                'file': test_file,
                'description': description,
                'success': success,
                'return_code': result.returncode,
                'test_count': test_count,
                'passed_count': passed_count,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': None  # Will be calculated later
            }

            if success:
                logging.info(f"✅ {description} completed successfully")
                if test_count > 0:
                    logging.info(f"   Tests: {passed_count}/{test_count} passed")
            else:
                logging.error(f"❌ {description} failed with return code {result.returncode}")
                if result.stderr:
                    logging.error(f"   Error: {result.stderr.strip()}")

            return result_info

        except subprocess.TimeoutExpired:
            logging.error(f"⏰ {description} timed out after 5 minutes")
            return {
                'file': test_file,
                'description': description,
                'success': False,
                'return_code': -1,
                'test_count': 0,
                'passed_count': 0,
                'stdout': '',
                'stderr': 'Test timed out',
                'duration': None
            }
        except Exception as e:
            logging.error(f"💥 {description} crashed: {e}")
            return {
                'file': test_file,
                'description': description,
                'success': False,
                'return_code': -1,
                'test_count': 0,
                'passed_count': 0,
                'stdout': '',
                'stderr': str(e),
                'duration': None
            }

    def run_all_tests(self):
        """Run all available tests"""
        logging.info("🚀 Starting DAO Test Suite")
        logging.info("=" * 60)

        # Define test files and descriptions
        tests = [
            ("test_statistical_optimization.py", "Statistical Optimization Pipeline"),
            ("tests/prog/test_dao.py", "DAO Core Functionality"),
        ]

        # Run each test
        for test_file, description in tests:
            if os.path.exists(test_file):
                result = self.run_test_file(test_file, description)
                self.test_results.append(result)
            else:
                logging.warning(f"⚠️  Test file {test_file} not found, skipping")

        # Calculate durations
        end_time = datetime.datetime.now()
        for result in self.test_results:
            result['duration'] = (end_time - self.start_time).total_seconds()

        # Generate summary
        self.generate_summary()

        return self.test_results

    def generate_summary(self):
        """Generate test summary report"""
        logging.info("\n" + "=" * 60)
        logging.info("📊 TEST SUITE SUMMARY")
        logging.info("=" * 60)

        total_tests = 0
        total_passed = 0
        total_failed = 0

        for result in self.test_results:
            test_count = result['test_count']
            passed_count = result['passed_count']

            total_tests += test_count
            total_passed += passed_count
            total_failed += (test_count - passed_count)

            status = "✅ PASS" if result['success'] else "❌ FAIL"
            logging.info(f"{status}: {result['description']}")

            if test_count > 0:
                success_rate = (passed_count / test_count) * 100
                logging.info(f"   Tests: {passed_count}/{test_count} passed ({success_rate:.1f}%)")

            if result['duration']:
                logging.info(f"   Duration: {result['duration']:.1f}s")

        # Overall summary
        logging.info("\n" + "-" * 40)
        if total_tests > 0:
            overall_success_rate = (total_passed / total_tests) * 100
            logging.info(f"Overall: {total_passed}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        else:
            logging.info("No tests were executed")

        # Success/failure count
        successful_runs = sum(1 for r in self.test_results if r['success'])
        total_runs = len(self.test_results)

        if total_runs > 0:
            run_success_rate = (successful_runs / total_runs) * 100
            logging.info(f"Test files: {successful_runs}/{total_runs} successful ({run_success_rate:.1f}%)")

        # Final status
        if successful_runs == total_runs and total_passed == total_tests:
            logging.info("\n🎉 All tests passed successfully!")
        elif successful_runs == total_runs:
            logging.info("\n⚠️  All test files ran but some individual tests failed")
        else:
            logging.info("\n❌ Some test files failed to run")

        # Save detailed report
        self.save_detailed_report()

    def save_detailed_report(self):
        """Save detailed test report to file"""
        report_file = f"test_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("DAO Test Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.datetime.now()}\n")
                f.write(f"Duration: {(datetime.datetime.now() - self.start_time).total_seconds():.1f}s\n\n")

                for result in self.test_results:
                    f.write(f"Test: {result['description']}\n")
                    f.write(f"File: {result['file']}\n")
                    f.write(f"Status: {'PASS' if result['success'] else 'FAIL'}\n")
                    f.write(f"Tests: {result['passed_count']}/{result['test_count']} passed\n")
                    f.write(f"Return Code: {result['return_code']}\n")

                    if result['stdout']:
                        f.write("\nSTDOUT:\n")
                        f.write(result['stdout'])
                        f.write("\n")

                    if result['stderr']:
                        f.write("\nSTDERR:\n")
                        f.write(result['stderr'])
                        f.write("\n")

                    f.write("-" * 30 + "\n\n")

            logging.info(f"📄 Detailed report saved to: {report_file}")

        except Exception as e:
            logging.error(f"Failed to save detailed report: {e}")

def main():
    """Main function"""
    runner = TestRunner()
    results = runner.run_all_tests()

    # Exit with appropriate code
    all_successful = all(r['success'] for r in results)
    sys.exit(0 if all_successful else 1)

if __name__ == "__main__":
    main()
