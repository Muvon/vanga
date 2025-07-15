#!/usr/bin/env python3
"""
VANGA Optimizer Performance Benchmarking Script

This script compares the performance of all 9 available optimizers on cryptocurrency data.
It runs training with each optimizer configuration and collects performance metrics.

Usage:
    python scripts/benchmark_optimizers.py --data data/BTCUSDT_1h.csv --symbol BTCUSDT
    python scripts/benchmark_optimizers.py --data data/ --batch  # Benchmark on multiple datasets
    python scripts/benchmark_optimizers.py --quick  # Quick benchmark with reduced epochs
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class BenchmarkResult:
    """Results from a single optimizer benchmark run"""
    optimizer: str
    symbol: str
    config_file: str

    # Training metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    epochs_trained: int
    training_time_seconds: float

    # Convergence metrics
    epochs_to_convergence: int
    convergence_stability: float  # Std dev of last 10 epochs

    # Performance metrics
    mae: float
    mse: float
    rmse: float
    mape: float
    sharpe_ratio: Optional[float]

    # Resource usage
    peak_memory_mb: float
    avg_epoch_time_seconds: float

    # Status
    converged: bool
    early_stopped: bool
    error_message: Optional[str]

class OptimizerBenchmark:
    """Main benchmarking class for VANGA optimizers"""

    def __init__(self, data_path: str, symbol: str, output_dir: str = "benchmark_results"):
        self.data_path = Path(data_path)
        self.symbol = symbol
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Available optimizer configurations
        self.optimizer_configs = {
            "AdamW": "configs/optimizer_examples/adamw_crypto_optimized.toml",
            "SGD": "configs/optimizer_examples/sgd_fine_tuning.toml",
            "Adam": "configs/optimizer_examples/adam_general_purpose.toml",
            "RMSprop": "configs/optimizer_examples/rmsprop_volatile_markets.toml",
            "NAdam": "configs/optimizer_examples/nadam_momentum_markets.toml",
            "RAdam": "configs/optimizer_examples/radam_stable_convergence.toml",
            "AdaMax": "configs/optimizer_examples/adamax_large_gradients.toml",
            "AdaDelta": "configs/optimizer_examples/adadelta_sparse_data.toml",
            "AdaGrad": "configs/optimizer_examples/adagrad_short_training.toml",
        }

        self.results: List[BenchmarkResult] = []

    async def run_single_benchmark(self, optimizer: str, config_file: str, quick_mode: bool = False) -> BenchmarkResult:
        """Run benchmark for a single optimizer"""
        print(f"🔄 Benchmarking {optimizer} optimizer...")

        # Create temporary config for benchmarking
        temp_config = self.create_benchmark_config(config_file, quick_mode)

        start_time = time.time()

        try:
            # Run VANGA training
            cmd = [
                "cargo", "run", "--release", "--",
                "train",
                "--symbol", self.symbol,
                "--data", str(self.data_path),
                "--config", temp_config,
                "--fresh"  # Always start fresh for fair comparison
            ]

            print(f"   Running: {' '.join(cmd)}")

            # Run with timeout and capture output
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd()
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=3600 if not quick_mode else 600  # 1 hour / 10 minutes timeout
            )

            training_time = time.time() - start_time

            if process.returncode == 0:
                # Parse training output for metrics
                output_text = stdout.decode() + stderr.decode()
                result = self.parse_training_output(
                    optimizer, config_file, output_text, training_time
                )
                print(f"   ✅ {optimizer}: {result.final_val_loss:.6f} val_loss in {result.epochs_trained} epochs")
                return result
            else:
                error_msg = stderr.decode()
                print(f"   ❌ {optimizer} failed: {error_msg[:200]}...")
                return self.create_error_result(optimizer, config_file, error_msg, training_time)

        except asyncio.TimeoutError:
            print(f"   ⏰ {optimizer} timed out")
            return self.create_error_result(optimizer, config_file, "Training timeout", time.time() - start_time)
        except Exception as e:
            print(f"   💥 {optimizer} crashed: {str(e)}")
            return self.create_error_result(optimizer, config_file, str(e), time.time() - start_time)
        finally:
            # Cleanup temporary config
            if os.path.exists(temp_config):
                os.remove(temp_config)

    def create_benchmark_config(self, base_config: str, quick_mode: bool) -> str:
        """Create a temporary config file optimized for benchmarking"""
        import toml

        # Load base configuration
        with open(base_config, 'r') as f:
            config = toml.load(f)

        # Modify for benchmarking
        if quick_mode:
            # Reduce epochs for quick benchmarking
            config['training']['epochs'] = min(config['training'].get('epochs', 100), 30)
            config['training']['early_stopping']['patience'] = 5

        # Ensure consistent validation split
        config['training']['validation_split'] = 0.2
        config['training']['shuffle_training'] = False

        # Add benchmarking-specific settings
        config['training']['save_model'] = False  # Don't save models during benchmarking

        # Create temporary config file
        temp_config = f"temp_benchmark_{int(time.time())}.toml"
        with open(temp_config, 'w') as f:
            toml.dump(config, f)

        return temp_config

    def parse_training_output(self, optimizer: str, config_file: str, output: str, training_time: float) -> BenchmarkResult:
        """Parse VANGA training output to extract metrics"""
        lines = output.split('\n')

        # Initialize default values
        final_train_loss = float('inf')
        final_val_loss = float('inf')
        best_val_loss = float('inf')
        epochs_trained = 0
        epochs_to_convergence = 0
        mae = float('inf')
        mse = float('inf')
        rmse = float('inf')
        mape = float('inf')
        converged = False
        early_stopped = False

        # Parse training logs
        val_losses = []
        epoch_times = []

        for line in lines:
            line = line.strip()

            # Parse epoch information
            if "Epoch" in line and "Loss:" in line:
                try:
                    # Extract epoch number
                    if "Epoch " in line:
                        epoch_part = line.split("Epoch ")[1].split("/")[0]
                        epochs_trained = max(epochs_trained, int(epoch_part))

                    # Extract training loss
                    if "Train Loss:" in line:
                        train_loss = float(line.split("Train Loss:")[1].split()[0])
                        final_train_loss = train_loss

                    # Extract validation loss
                    if "Val Loss:" in line:
                        val_loss = float(line.split("Val Loss:")[1].split()[0])
                        final_val_loss = val_loss
                        val_losses.append(val_loss)
                        best_val_loss = min(best_val_loss, val_loss)

                except (ValueError, IndexError):
                    continue

            # Parse evaluation metrics
            if "MAE:" in line:
                try:
                    mae = float(line.split("MAE:")[1].split()[0])
                except (ValueError, IndexError):
                    pass

            if "MSE:" in line:
                try:
                    mse = float(line.split("MSE:")[1].split()[0])
                    rmse = mse ** 0.5
                except (ValueError, IndexError):
                    pass

            if "MAPE:" in line:
                try:
                    mape = float(line.split("MAPE:")[1].split()[0].rstrip('%'))
                except (ValueError, IndexError):
                    pass

            # Check for early stopping
            if "Early stopping" in line:
                early_stopped = True

            # Check for convergence
            if "Training completed" in line or "Model saved" in line:
                converged = True

        # Calculate convergence metrics
        if len(val_losses) > 10:
            # Find convergence point (when validation loss stabilizes)
            for i in range(10, len(val_losses)):
                recent_losses = val_losses[i-10:i]
                if len(recent_losses) >= 5:
                    std_dev = pd.Series(recent_losses).std()
                    if std_dev < 0.001:  # Converged when std dev is small
                        epochs_to_convergence = i
                        break

            convergence_stability = pd.Series(val_losses[-10:]).std()
        else:
            convergence_stability = float('inf')

        # Calculate average epoch time
        avg_epoch_time = training_time / max(epochs_trained, 1)

        return BenchmarkResult(
            optimizer=optimizer,
            symbol=self.symbol,
            config_file=config_file,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            epochs_trained=epochs_trained,
            training_time_seconds=training_time,
            epochs_to_convergence=epochs_to_convergence or epochs_trained,
            convergence_stability=convergence_stability,
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            sharpe_ratio=None,  # Would need additional calculation
            peak_memory_mb=0.0,  # Would need memory monitoring
            avg_epoch_time_seconds=avg_epoch_time,
            converged=converged,
            early_stopped=early_stopped,
            error_message=None
        )

    def create_error_result(self, optimizer: str, config_file: str, error_msg: str, training_time: float) -> BenchmarkResult:
        """Create a result object for failed runs"""
        return BenchmarkResult(
            optimizer=optimizer,
            symbol=self.symbol,
            config_file=config_file,
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            best_val_loss=float('inf'),
            epochs_trained=0,
            training_time_seconds=training_time,
            epochs_to_convergence=0,
            convergence_stability=float('inf'),
            mae=float('inf'),
            mse=float('inf'),
            rmse=float('inf'),
            mape=float('inf'),
            sharpe_ratio=None,
            peak_memory_mb=0.0,
            avg_epoch_time_seconds=0.0,
            converged=False,
            early_stopped=False,
            error_message=error_msg
        )

    async def run_full_benchmark(self, quick_mode: bool = False, optimizers: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """Run benchmark for all optimizers"""
        print(f"🚀 Starting VANGA Optimizer Benchmark")
        print(f"   Symbol: {self.symbol}")
        print(f"   Data: {self.data_path}")
        print(f"   Mode: {'Quick' if quick_mode else 'Full'}")
        print(f"   Output: {self.output_dir}")
        print()

        # Filter optimizers if specified
        configs_to_test = self.optimizer_configs
        if optimizers:
            configs_to_test = {k: v for k, v in configs_to_test.items() if k in optimizers}

        # Run benchmarks sequentially to avoid resource conflicts
        results = []
        for optimizer, config_file in configs_to_test.items():
            if not os.path.exists(config_file):
                print(f"⚠️  Config file not found: {config_file}")
                continue

            result = await self.run_single_benchmark(optimizer, config_file, quick_mode)
            results.append(result)

            # Brief pause between runs
            await asyncio.sleep(2)

        self.results = results
        return results

    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        if not self.results:
            return "No benchmark results available."

        # Sort results by validation loss (best first)
        successful_results = [r for r in self.results if r.converged and r.error_message is None]
        failed_results = [r for r in self.results if r.error_message is not None]

        successful_results.sort(key=lambda x: x.best_val_loss)

        report = []
        report.append("# VANGA Optimizer Benchmark Report")
        report.append(f"**Symbol:** {self.symbol}")
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Summary table
        report.append("## 🏆 Performance Ranking")
        report.append("")
        report.append("| Rank | Optimizer | Best Val Loss | Epochs | Time (s) | Converged | Status |")
        report.append("|------|-----------|---------------|--------|----------|-----------|--------|")

        for i, result in enumerate(successful_results, 1):
            status = "✅ Success"
            if result.early_stopped:
                status = "⏹️ Early Stop"

            report.append(f"| {i} | **{result.optimizer}** | {result.best_val_loss:.6f} | {result.epochs_trained} | {result.training_time_seconds:.1f} | {result.converged} | {status} |")

        # Failed runs
        if failed_results:
            report.append("")
            report.append("## ❌ Failed Runs")
            report.append("")
            for result in failed_results:
                report.append(f"- **{result.optimizer}**: {result.error_message}")

        # Detailed analysis
        if successful_results:
            best_result = successful_results[0]
            report.append("")
            report.append("## 📊 Detailed Analysis")
            report.append("")
            report.append(f"### 🥇 Best Performer: {best_result.optimizer}")
            report.append(f"- **Validation Loss:** {best_result.best_val_loss:.6f}")
            report.append(f"- **Training Time:** {best_result.training_time_seconds:.1f} seconds")
            report.append(f"- **Epochs to Convergence:** {best_result.epochs_to_convergence}")
            report.append(f"- **Convergence Stability:** {best_result.convergence_stability:.6f}")
            report.append(f"- **MAE:** {best_result.mae:.6f}")
            report.append(f"- **RMSE:** {best_result.rmse:.6f}")
            report.append("")

            # Performance comparison
            report.append("### 📈 Performance Metrics")
            report.append("")
            report.append("| Optimizer | Val Loss | MAE | RMSE | MAPE | Epochs | Time (s) |")
            report.append("|-----------|----------|-----|------|------|--------|----------|")

            for result in successful_results:
                report.append(f"| {result.optimizer} | {result.best_val_loss:.6f} | {result.mae:.6f} | {result.rmse:.6f} | {result.mape:.2f}% | {result.epochs_trained} | {result.training_time_seconds:.1f} |")

        # Recommendations
        report.append("")
        report.append("## 💡 Recommendations")
        report.append("")

        if successful_results:
            best = successful_results[0]
            report.append(f"1. **Primary Choice:** Use **{best.optimizer}** for best performance")

            # Find fastest converging
            fastest = min(successful_results, key=lambda x: x.epochs_to_convergence)
            if fastest.optimizer != best.optimizer:
                report.append(f"2. **Fast Training:** Use **{fastest.optimizer}** for quickest convergence ({fastest.epochs_to_convergence} epochs)")

            # Find most stable
            most_stable = min(successful_results, key=lambda x: x.convergence_stability)
            if most_stable.optimizer not in [best.optimizer, fastest.optimizer]:
                report.append(f"3. **Stable Training:** Use **{most_stable.optimizer}** for most stable convergence")

        return "\\n".join(report)

    def save_results(self):
        """Save benchmark results to files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save detailed results as JSON
        results_data = []
        for result in self.results:
            results_data.append({
                'optimizer': result.optimizer,
                'symbol': result.symbol,
                'config_file': result.config_file,
                'final_train_loss': result.final_train_loss,
                'final_val_loss': result.final_val_loss,
                'best_val_loss': result.best_val_loss,
                'epochs_trained': result.epochs_trained,
                'training_time_seconds': result.training_time_seconds,
                'epochs_to_convergence': result.epochs_to_convergence,
                'convergence_stability': result.convergence_stability,
                'mae': result.mae,
                'mse': result.mse,
                'rmse': result.rmse,
                'mape': result.mape,
                'converged': result.converged,
                'early_stopped': result.early_stopped,
                'error_message': result.error_message
            })

        json_file = self.output_dir / f"benchmark_results_{self.symbol}_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save report as markdown
        report = self.generate_report()
        report_file = self.output_dir / f"benchmark_report_{self.symbol}_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)

        print(f"📊 Results saved:")
        print(f"   JSON: {json_file}")
        print(f"   Report: {report_file}")

        return json_file, report_file

    def create_visualizations(self):
        """Create performance visualization charts"""
        if not self.results:
            return

        successful_results = [r for r in self.results if r.converged and r.error_message is None]
        if not successful_results:
            return

        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'VANGA Optimizer Benchmark Results - {self.symbol}', fontsize=16)

        # 1. Validation Loss Comparison
        optimizers = [r.optimizer for r in successful_results]
        val_losses = [r.best_val_loss for r in successful_results]

        axes[0, 0].bar(optimizers, val_losses, color='skyblue')
        axes[0, 0].set_title('Best Validation Loss by Optimizer')
        axes[0, 0].set_ylabel('Validation Loss')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 2. Training Time Comparison
        training_times = [r.training_time_seconds for r in successful_results]

        axes[0, 1].bar(optimizers, training_times, color='lightcoral')
        axes[0, 1].set_title('Training Time by Optimizer')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 3. Epochs to Convergence
        epochs_to_conv = [r.epochs_to_convergence for r in successful_results]

        axes[1, 0].bar(optimizers, epochs_to_conv, color='lightgreen')
        axes[1, 0].set_title('Epochs to Convergence')
        axes[1, 0].set_ylabel('Epochs')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 4. Performance vs Speed Scatter
        axes[1, 1].scatter(training_times, val_losses, s=100, alpha=0.7)
        for i, opt in enumerate(optimizers):
            axes[1, 1].annotate(opt, (training_times[i], val_losses[i]),
                              xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Best Validation Loss')
        axes[1, 1].set_title('Performance vs Speed Trade-off')

        plt.tight_layout()

        # Save the plot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"benchmark_plots_{self.symbol}_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Visualization saved: {plot_file}")
        return plot_file

async def main():
    parser = argparse.ArgumentParser(description="VANGA Optimizer Performance Benchmark")
    parser.add_argument("--data", required=True, help="Path to CSV data file or directory")
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("--output", default="benchmark_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark with reduced epochs")
    parser.add_argument("--optimizers", nargs="+", help="Specific optimizers to test",
                       choices=["AdamW", "SGD", "Adam", "RMSprop", "NAdam", "RAdam", "AdaMax", "AdaDelta", "AdaGrad"])
    parser.add_argument("--batch", action="store_true", help="Batch mode for multiple datasets")

    args = parser.parse_args()

    # Validate data path
    if not os.path.exists(args.data):
        print(f"❌ Data path not found: {args.data}")
        sys.exit(1)

    # Run benchmark
    benchmark = OptimizerBenchmark(args.data, args.symbol, args.output)

    try:
        results = await benchmark.run_full_benchmark(
            quick_mode=args.quick,
            optimizers=args.optimizers
        )

        # Generate and save results
        json_file, report_file = benchmark.save_results()

        # Create visualizations
        try:
            plot_file = benchmark.create_visualizations()
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")

        # Print summary
        print()
        print("🎉 Benchmark completed!")
        print()
        print(benchmark.generate_report())

    except KeyboardInterrupt:
        print("\\n⏹️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Benchmark failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
