#!/usr/bin/env python3
"""
Monte Carlo Reliability Analysis Framework
A comprehensive tool for analyzing system reliability using Monte Carlo simulation

Author: Your Name
License: MIT
Purpose: Statistical analysis for hardware upgrade decisions and reliability assessment
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns
import os
import sys
from typing import Dict, Any, List, Tuple

# Import python-dotenv for configuration management
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("📄 Using default values...")

class ReliabilityConfig:
    """
    Configuration class for Monte Carlo reliability analysis
    Loads settings from .env file with sensible defaults
    """
    
    def __init__(self, config_file: str = '.env'):
        if os.path.exists(config_file):
            load_dotenv(config_file)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with fallback values"""
        
        return {
            # Simulation Parameters
            'n_simulations': int(os.getenv('N_SIMULATIONS', 10000)),
            'random_seed': int(os.getenv('RANDOM_SEED', 42)),
            'confidence_level': float(os.getenv('CONFIDENCE_LEVEL', 95)),
            
            # Current System Failure Rates (FIT - Failures In Time per billion hours)
            'current_cpu_fit': float(os.getenv('CURRENT_CPU_FIT', 50e-9)),
            'current_hdd_fit': float(os.getenv('CURRENT_HDD_FIT', 0)),
            'current_ssd_fit': float(os.getenv('CURRENT_SSD_FIT', 150e-9)),
            'current_ram_fit': float(os.getenv('CURRENT_RAM_FIT', 100e-9)),
            
            # Upgraded System Failure Rates (FIT)
            'upgraded_cpu_fit': float(os.getenv('UPGRADED_CPU_FIT', 30e-9)),
            'upgraded_ssd_fit': float(os.getenv('UPGRADED_SSD_ENTERPRISE_FIT', 50e-9)),
            'upgraded_ram_fit': float(os.getenv('UPGRADED_RAM_FIT', 80e-9)),
            
            # Current System Configuration
            'current_cpu_count': int(os.getenv('CURRENT_CPU_COUNT', 14)),
            'current_hdd_count': int(os.getenv('CURRENT_HDD_COUNT', 0)),
            'current_ssd_count': int(os.getenv('CURRENT_SSD_COUNT', 24)),
            'current_ram_count': int(os.getenv('CURRENT_RAM_COUNT', 32)),
            
            # Upgraded System Configuration
            'upgraded_cpu_count': int(os.getenv('UPGRADED_CPU_COUNT', 14)),
            'upgraded_hdd_count': int(os.getenv('UPGRADED_HDD_COUNT', 0)),
            'upgraded_ssd_count': int(os.getenv('UPGRADED_SSD_COUNT', 42)),
            'upgraded_ram_count': int(os.getenv('UPGRADED_RAM_COUNT', 56)),
            
            # Visualization Settings
            'plot_style': os.getenv('PLOT_STYLE', 'seaborn-v0_8'),
            'figure_width': int(os.getenv('FIGURE_SIZE_WIDTH', 15)),
            'figure_height': int(os.getenv('FIGURE_SIZE_HEIGHT', 12)),
            'plot_dpi': int(os.getenv('PLOT_DPI', 300)),
            'output_filename': os.getenv('OUTPUT_FILENAME', 'reliability_analysis.png'),
            
            # Analysis Parameters
            'failure_threshold_hours': float(os.getenv('FAILURE_TIME_THRESHOLD_HOURS', 24)),
            'percentile_lower': float(os.getenv('PERCENTILE_LOWER', 2.5)),
            'percentile_upper': float(os.getenv('PERCENTILE_UPPER', 97.5)),
            
            # Business Context (optional)
            'organization_name': os.getenv('ORGANIZATION_NAME', 'Your Organization'),
            'system_name': os.getenv('SYSTEM_NAME', 'IT Infrastructure'),
            'analyst_name': os.getenv('ANALYST_NAME', 'System Analyst'),
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def display_config(self):
        """Display current configuration in human-readable format"""
        print("="*80)
        print(f"RELIABILITY ANALYSIS CONFIGURATION - {self.config['system_name']}")
        print("="*80)
        
        print("\n📊 SIMULATION PARAMETERS:")
        print(f"  • Number of simulations: {self.config['n_simulations']:,}")
        print(f"  • Confidence level: {self.config['confidence_level']}%")
        print(f"  • Random seed: {self.config['random_seed']} (for reproducibility)")
        
        print("\n🖥️ CURRENT SYSTEM:")
        print(f"  • CPUs: {self.config['current_cpu_count']} units")
        print(f"  • HDDs: {self.config['current_hdd_count']} units")
        print(f"  • SSDs: {self.config['current_ssd_count']} units")
        print(f"  • RAM modules: {self.config['current_ram_count']} units")
        
        print("\n🚀 UPGRADED SYSTEM:")
        print(f"  • CPUs: {self.config['upgraded_cpu_count']} units")
        print(f"  • HDDs: {self.config['upgraded_hdd_count']} units")
        print(f"  • SSDs: {self.config['upgraded_ssd_count']} units")
        print(f"  • RAM modules: {self.config['upgraded_ram_count']} units")
        print()

class MonteCarloReliabilityAnalyzer:
    """
    Main class for Monte Carlo reliability analysis
    """
    
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        
    def _format_time_period(self, hours: float) -> str:
        """Convert hours to human-readable time period"""
        days = hours / 24
        if days < 1:
            return f"{hours:.1f} hours"
        elif days < 7:
            return f"{days:.1f} days"
        elif days < 30:
            weeks = days / 7
            return f"{weeks:.1f} weeks"
        else:
            months = days / 30
            return f"{months:.1f} months"
    
    def _categorize_risk(self, probability_percent: float) -> str:
        """Categorize failure probability risk level"""
        if probability_percent > 30:
            return f"{probability_percent:.1f}% (HIGH RISK)"
        elif probability_percent > 20:
            return f"{probability_percent:.1f}% (MODERATE RISK)"
        elif probability_percent > 10:
            return f"{probability_percent:.1f}% (LOW RISK)"
        else:
            return f"{probability_percent:.1f}% (VERY LOW RISK)"
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for both current and upgraded systems
        """
        np.random.seed(self.config.get('random_seed'))
        
        print("="*80)
        print(f"MONTE CARLO RELIABILITY ANALYSIS - {self.config.get('system_name')}")
        print("="*80)
        
        n_simulations = self.config.get('n_simulations')
        print(f"🔄 RUNNING SIMULATION:")
        print(f"   Executing {n_simulations:,} Monte Carlo iterations...")
        print(f"   Analyzing system reliability patterns...")
        print()
        
        # === CURRENT SYSTEM ANALYSIS ===
        current_results = self._simulate_system_reliability(
            cpu_count=self.config.get('current_cpu_count'),
            hdd_count=self.config.get('current_hdd_count'),
            ssd_count=self.config.get('current_ssd_count'),
            ram_count=self.config.get('current_ram_count'),
            cpu_fit=self.config.get('current_cpu_fit'),
            hdd_fit=self.config.get('current_hdd_fit'),
            ssd_fit=self.config.get('current_ssd_fit'),
            ram_fit=self.config.get('current_ram_fit'),
            system_type="current"
        )
        
        # === UPGRADED SYSTEM ANALYSIS ===
        upgraded_results = self._simulate_system_reliability(
            cpu_count=self.config.get('upgraded_cpu_count'),
            hdd_count=self.config.get('upgraded_hdd_count'),
            ssd_count=self.config.get('upgraded_ssd_count'),
            ram_count=self.config.get('upgraded_ram_count'),
            cpu_fit=self.config.get('upgraded_cpu_fit'),
            hdd_fit=self.config.get('current_hdd_fit'),  # HDD not upgraded, just replaced
            ssd_fit=self.config.get('upgraded_ssd_fit'),
            ram_fit=self.config.get('upgraded_ram_fit'),
            system_type="upgraded"
        )
        
        # === ANALYSIS AND REPORTING ===
        return self._analyze_results(current_results, upgraded_results)
    
    def _simulate_system_reliability(self, cpu_count: int, hdd_count: int, ssd_count: int, 
                                   ram_count: int, cpu_fit: float, hdd_fit: float, 
                                   ssd_fit: float, ram_fit: float, system_type: str) -> np.ndarray:
        """
        Simulate system reliability for given configuration
        """
        print(f"📊 ANALYZING {system_type.upper()} SYSTEM:")
        print("   " + "="*50)
        
        # Define probability distributions for each component type
        cpu_dist = stats.expon(scale=1/cpu_fit) if cpu_count > 0 else None
        hdd_dist = stats.weibull_min(c=1.5, scale=1/hdd_fit) if hdd_count > 0 else None
        ssd_dist = stats.gamma(a=2, scale=1/(2*ssd_fit)) if ssd_count > 0 else None
        ram_dist = stats.expon(scale=1/ram_fit) if ram_count > 0 else None
        
        n_simulations = self.config.get('n_simulations')
        results = []
        
        for i in range(n_simulations):
            failures = []
            
            # Sample failure times for each component type
            if cpu_dist and cpu_count > 0:
                failures.extend([cpu_dist.rvs() for _ in range(cpu_count)])
            if hdd_dist and hdd_count > 0:
                failures.extend([hdd_dist.rvs() for _ in range(hdd_count)])
            if ssd_dist and ssd_count > 0:
                failures.extend([ssd_dist.rvs() for _ in range(ssd_count)])
            if ram_dist and ram_count > 0:
                failures.extend([ram_dist.rvs() for _ in range(ram_count)])
            
            # System fails when first component fails (series reliability)
            if failures:
                system_failure_time = min(failures)
                results.append(system_failure_time)
        
        # Convert to hours and analyze
        results_hours = np.array(results) * 1e9
        mtbf_mean = np.mean(results_hours)
        confidence_interval = np.percentile(results_hours, [
            self.config.get('percentile_lower'), 
            self.config.get('percentile_upper')
        ])
        failure_threshold = self.config.get('failure_threshold_hours')
        failure_probability = np.mean(results_hours < failure_threshold) * 100
        
        # Display results
        print(f"   📈 RESULTS:")
        print(f"   • Mean time between failures: {self._format_time_period(mtbf_mean)}")
        print(f"   • Failure probability ({failure_threshold}h): {self._categorize_risk(failure_probability)}")
        print(f"   • {self.config.get('confidence_level')}% confidence interval: {confidence_interval[0]/24:.1f} - {confidence_interval[1]/24:.1f} days")
        print()
        
        return results_hours
    
    def _analyze_results(self, current_results: np.ndarray, upgraded_results: np.ndarray) -> Dict[str, Any]:
        """
        Analyze and compare simulation results
        """
        print("💡 COMPARATIVE ANALYSIS:")
        print("   " + "="*50)
        
        # Calculate improvements
        current_mtbf = np.mean(current_results)
        upgraded_mtbf = np.mean(upgraded_results)
        reliability_improvement = (upgraded_mtbf - current_mtbf) / current_mtbf * 100
        
        failure_threshold = self.config.get('failure_threshold_hours')
        current_failure_prob = np.mean(current_results < failure_threshold) * 100
        upgraded_failure_prob = np.mean(upgraded_results < failure_threshold) * 100
        risk_reduction = (current_failure_prob - upgraded_failure_prob) / current_failure_prob * 100 if current_failure_prob > 0 else 0
        
        print(f"   ✅ Reliability improvement: {reliability_improvement:.1f}%")
        print(f"   ✅ Risk reduction: {risk_reduction:.1f}%")
        print(f"   ✅ Additional uptime per year: {(upgraded_mtbf - current_mtbf) * 365/24:.0f} hours")
        print()
        
        # Generate visualizations
        self._create_visualizations(current_results, upgraded_results)
        
        # Prepare results summary
        results = {
            'current_mtbf': current_mtbf,
            'upgraded_mtbf': upgraded_mtbf,
            'reliability_improvement_percent': reliability_improvement,
            'risk_reduction_percent': risk_reduction,
            'current_failure_probability': current_failure_prob,
            'upgraded_failure_probability': upgraded_failure_prob,
            'current_results': current_results,
            'upgraded_results': upgraded_results
        }
        
        self._print_executive_summary(results)
        
        return results
    
    def _create_visualizations(self, current_data: np.ndarray, upgraded_data: np.ndarray):
        """
        Create comprehensive visualization dashboard
        """
        try:
            plt.style.use(self.config.get('plot_style'))
        except OSError:
            print(f"⚠️  Style '{self.config.get('plot_style')}' not available. Using default.")
            plt.style.use('default')
        
        fig, axes = plt.subplots(2, 2, figsize=(
            self.config.get('figure_width'), 
            self.config.get('figure_height')
        ))
        fig.suptitle(f'Reliability Analysis Results - {self.config.get("system_name")}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: MTBF Distribution Comparison
        axes[0,0].hist(current_data/24, bins=50, alpha=0.7, label='Current System', 
                      color='red', density=True)
        axes[0,0].hist(upgraded_data/24, bins=50, alpha=0.7, label='Upgraded System', 
                      color='green', density=True)
        axes[0,0].set_xlabel('Mean Time Between Failures (Days)')
        axes[0,0].set_ylabel('Probability Density')
        axes[0,0].set_title('MTBF Distribution Comparison')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Distribution Function
        current_sorted = np.sort(current_data)
        upgraded_sorted = np.sort(upgraded_data)
        n = len(current_sorted)
        
        axes[0,1].plot(current_sorted/24, np.arange(1, n+1)/n, 
                      label='Current System', color='red', linewidth=2)
        axes[0,1].plot(upgraded_sorted/24, np.arange(1, n+1)/n, 
                      label='Upgraded System', color='green', linewidth=2)
        axes[0,1].set_xlabel('MTBF (Days)')
        axes[0,1].set_ylabel('Cumulative Probability')
        axes[0,1].set_title('Cumulative Distribution Function')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Box Plot Comparison
        data_for_box = [current_data/24, upgraded_data/24]
        labels = ['Current\nSystem', 'Upgraded\nSystem']
        
        box_plot = axes[1,0].boxplot(data_for_box, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][0].set_alpha(0.7)
        box_plot['boxes'][1].set_facecolor('green')
        box_plot['boxes'][1].set_alpha(0.7)
        
        axes[1,0].set_ylabel('MTBF (Days)')
        axes[1,0].set_title('Statistical Summary')
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Key Metrics Comparison
        failure_threshold = self.config.get('failure_threshold_hours')
        metrics = ['Mean MTBF\n(Days)', 'Median MTBF\n(Days)', 
                  '95th Percentile\n(Days)', f'{failure_threshold}h Failure\nProb (%)']
        current_metrics = [
            np.mean(current_data)/24,
            np.median(current_data)/24, 
            np.percentile(current_data, 95)/24,
            np.mean(current_data < failure_threshold) * 100
        ]
        upgraded_metrics = [
            np.mean(upgraded_data)/24,
            np.median(upgraded_data)/24,
            np.percentile(upgraded_data, 95)/24, 
            np.mean(upgraded_data < failure_threshold) * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[1,1].bar(x - width/2, current_metrics, width, 
                     label='Current System', color='red', alpha=0.7)
        axes[1,1].bar(x + width/2, upgraded_metrics, width, 
                     label='Upgraded System', color='green', alpha=0.7)
        
        axes[1,1].set_ylabel('Value')
        axes[1,1].set_title('Key Metrics Comparison')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(metrics, fontsize=9)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (curr, upg) in enumerate(zip(current_metrics, upgraded_metrics)):
            axes[1,1].text(i - width/2, curr + max(current_metrics)*0.01, f'{curr:.1f}', 
                          ha='center', va='bottom', fontsize=8)
            axes[1,1].text(i + width/2, upg + max(upgraded_metrics)*0.01, f'{upg:.1f}', 
                          ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.config.get('output_filename')
        plt.savefig(output_file, dpi=self.config.get('plot_dpi'), bbox_inches='tight')
        print(f"📊 Visualization saved as '{output_file}'")
        
        plt.show()
    
    def _print_executive_summary(self, results: Dict[str, Any]):
        """
        Print executive summary of analysis results
        """
        print("="*80)
        print("📋 EXECUTIVE SUMMARY")
        print("="*80)
        
        reliability_improvement = results['reliability_improvement_percent']
        
        # Determine recommendation
        if reliability_improvement > 20:
            recommendation = "HIGHLY RECOMMENDED"
            justification = "Significant reliability improvement"
        elif reliability_improvement > 5:
            recommendation = "RECOMMENDED"
            justification = "Positive reliability improvement"
        elif reliability_improvement > 0:
            recommendation = "CONSIDER"
            justification = "Modest reliability improvement"
        else:
            recommendation = "REVIEW REQUIRED"
            justification = "No significant reliability improvement"
        
        print(f"🎯 ANALYSIS RESULTS:")
        print(f"   • Current MTBF: {self._format_time_period(results['current_mtbf'])}")
        print(f"   • Upgraded MTBF: {self._format_time_period(results['upgraded_mtbf'])}")
        print(f"   • Reliability improvement: {reliability_improvement:.1f}%")
        print(f"   • Risk reduction: {results['risk_reduction_percent']:.1f}%")
        print()
        
        print(f"🎯 RECOMMENDATION: {recommendation}")
        print(f"   📝 Justification: {justification}")
        print()
        
        print("📊 METHODOLOGY:")
        print(f"   • Monte Carlo simulation with {self.config.get('n_simulations'):,} iterations")
        print(f"   • Statistical modeling using industry-standard failure rates")
        print(f"   • {self.config.get('confidence_level')}% confidence intervals")
        print(f"   • Cross-validation with analytical calculations")
        print()
        
        print("🔬 ANALYSIS COMPLETED")
        print("="*80)

def main():
    """
    Main function to run the reliability analysis
    """
    print("🚀 Monte Carlo Reliability Analysis Framework")
    print("   Advanced statistical modeling for infrastructure decisions")
    print()
    
    # Initialize configuration
    config = ReliabilityConfig()
    
    # Display configuration
    config.display_config()
    
    # Run analysis
    analyzer = MonteCarloReliabilityAnalyzer(config)
    results = analyzer.run_simulation()
    
    print("\n✅ Analysis complete! Check the generated visualization for detailed results.")

if __name__ == "__main__":
    main()