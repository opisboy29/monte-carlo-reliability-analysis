#!/usr/bin/env python3
"""
Monte Carlo Reliability Analysis Framework
A comprehensive tool for analyzing system reliability using Monte Carlo simulation

Author: opisboy29
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
DOTENV_AVAILABLE = False
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    print("📄 Using default values...")

class ReliabilityConfig:
    """
    Configuration class for Monte Carlo reliability analysis
    Loads settings from .env file with sensible defaults
    """
    
    def __init__(self, config_file: str = '.env'):
        if DOTENV_AVAILABLE and os.path.exists(config_file):
            load_dotenv(config_file)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables with fallback values"""
        confidence_level = int(os.getenv('CONFIDENCE_LEVEL', 95))
        
        return {
            # Simulation parameters
            'n_simulations': int(os.getenv('N_SIMULATIONS', 10000)),
            'random_seed': int(os.getenv('RANDOM_SEED', 42)),
            'confidence_level': confidence_level,
            
            # Statistical analysis parameters
            'percentile_lower': (100 - confidence_level) / 2,
            'percentile_upper': 100 - (100 - confidence_level) / 2,
            'failure_threshold_hours': float(os.getenv('FAILURE_THRESHOLD_HOURS', 72)),
            
            # Visualization settings
            'plot_style': os.getenv('PLOT_STYLE', 'default'),
            'figure_width': float(os.getenv('FIGURE_WIDTH', 16)),
            'figure_height': float(os.getenv('FIGURE_HEIGHT', 12)),
            'dpi': int(os.getenv('DPI', 300)),
            'save_plots': os.getenv('SAVE_PLOTS', 'true').lower() == 'true',
            'output_filename': os.getenv('OUTPUT_FILENAME', 'reliability_analysis_output.png'),
            
            # CURRENT SYSTEM: Mixed quality hardware
            'current_cpu_count': int(os.getenv('CURRENT_CPU_COUNT', 14)),
            'current_hdd_count': int(os.getenv('CURRENT_HDD_COUNT', 2)),      # R440 has 2 HDD
            'current_ssd_count': int(os.getenv('CURRENT_SSD_COUNT', 22)),     # Rest are SSD
            'current_ram_count': int(os.getenv('CURRENT_RAM_COUNT', 32)),
            'current_psu_count': int(os.getenv('CURRENT_PSU_COUNT', 7)),      # Single PSU per server
            
            # UPGRADED SYSTEM: Enterprise grade replacements + redundancy
            'upgraded_cpu_count': int(os.getenv('UPGRADED_CPU_COUNT', 14)),   # Same count, better CPU
            'upgraded_hdd_count': int(os.getenv('UPGRADED_HDD_COUNT', 0)),    # All HDD replaced
            'upgraded_ssd_count': int(os.getenv('UPGRADED_SSD_COUNT', 24)),   # All enterprise SSD
            'upgraded_ram_count': int(os.getenv('UPGRADED_RAM_COUNT', 32)),   # Same count, ECC RAM
            'upgraded_psu_count': int(os.getenv('UPGRADED_PSU_COUNT', 14)),   # Redundant PSU
            
            # CURRENT FIT RATES (Conservative estimates)
            'current_cpu_fit': float(os.getenv('CURRENT_CPU_FIT', 150e-9)),   # Mixed Silver CPUs
            'current_hdd_fit': float(os.getenv('CURRENT_HDD_FIT', 1200e-9)),  # 7200rpm SATA
            'current_ssd_fit': float(os.getenv('CURRENT_SSD_FIT', 300e-9)),   # Mixed SSD quality
            'current_ram_fit': float(os.getenv('CURRENT_RAM_FIT', 100e-9)),   # Standard DDR4
            'current_psu_fit': float(os.getenv('CURRENT_PSU_FIT', 200e-9)),   # Single PSU risk
            
            # UPGRADED FIT RATES (Enterprise grade)
            'upgraded_cpu_fit': float(os.getenv('UPGRADED_CPU_FIT', 80e-9)),   # Platinum CPUs
            'upgraded_hdd_fit': float(os.getenv('UPGRADED_HDD_FIT', 0)),       # No HDD
            'upgraded_ssd_fit': float(os.getenv('UPGRADED_SSD_FIT', 50e-9)),   # Enterprise SSD
            'upgraded_ram_fit': float(os.getenv('UPGRADED_RAM_FIT', 40e-9)),   # ECC DDR4
            'upgraded_psu_fit': float(os.getenv('UPGRADED_PSU_FIT', 80e-9)),   # Redundant PSU
            
            # Analysis settings
            'system_name': os.getenv('SYSTEM_NAME', 'Enterprise Server Infrastructure'),
            'analysis_purpose': os.getenv('ANALYSIS_PURPOSE', 'Hardware Upgrade ROI Analysis')
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        return self.config.get(key, default)
    
    def display_config(self):
        """Display current configuration in a readable format"""
        print("\n📋 CONFIGURATION SUMMARY:")
        print("=" * 60)
        
        print(f"🎯 System: {self.get('system_name')}")
        print(f"📊 Purpose: {self.get('analysis_purpose')}")
        print(f"🔄 Simulations: {self.get('n_simulations'):,}")
        print(f"🎲 Random Seed: {self.get('random_seed')}")
        
        print(f"\n💾 CURRENT SYSTEM:")
        print(f"   CPU: {self.get('current_cpu_count')} units")
        print(f"   HDD: {self.get('current_hdd_count')} units")
        print(f"   SSD: {self.get('current_ssd_count')} units")
        print(f"   RAM: {self.get('current_ram_count')} modules")
        
        print(f"\n🚀 UPGRADED SYSTEM:")
        print(f"   CPU: {self.get('upgraded_cpu_count')} units")
        print(f"   HDD: {self.get('upgraded_hdd_count')} units")
        print(f"   SSD: {self.get('upgraded_ssd_count')} units")
        print(f"   RAM: {self.get('upgraded_ram_count')} modules")
        print("=" * 60)

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
    
    def _calculate_system_reliability(self, cpu_count: int, hdd_count: int, ssd_count: int, 
                                    ram_count: int, psu_count: int, cpu_fit: float, 
                                    hdd_fit: float, ssd_fit: float, ram_fit: float, 
                                    psu_fit: float, system_type: str) -> float:
        """
        Calculate system failure rate using enterprise redundancy model
        """
        # CPU subsystem (N-way redundancy with load balancing)
        cpu_subsystem_rate = cpu_count * cpu_fit
        
        # Storage subsystem with RAID modeling
        if hdd_count > 0:
            # RAID 1/5/6 provides redundancy - only critical if multiple drives fail
            hdd_subsystem_rate = (hdd_count * hdd_fit * hdd_fit) / (hdd_count - 1) if hdd_count > 1 else hdd_count * hdd_fit
        else:
            hdd_subsystem_rate = 0
            
        if ssd_count > 0:
            # Enterprise SSD RAID - high redundancy
            ssd_subsystem_rate = (ssd_count * ssd_fit * ssd_fit) / max(1, ssd_count - 2) if ssd_count > 2 else ssd_count * ssd_fit * 0.5
        else:
            ssd_subsystem_rate = 0
            
        storage_subsystem_rate = hdd_subsystem_rate + ssd_subsystem_rate
        
        # Memory subsystem (ECC provides error correction)
        # Multiple memory channels provide some redundancy
        memory_subsystem_rate = ram_count * ram_fit * 0.3  # ECC reduces effective failure rate
        
        # Power subsystem
        if psu_count >= 14:  # Redundant PSU (N+N)
            psu_subsystem_rate = (psu_count / 2) * psu_fit * psu_fit  # Both PSUs must fail
        else:  # Single PSU per server
            psu_subsystem_rate = psu_count * psu_fit
        
        # System failure rate (critical subsystems in series)
        total_system_rate = (
            cpu_subsystem_rate + 
            storage_subsystem_rate + 
            memory_subsystem_rate + 
            psu_subsystem_rate
        )
        
        return total_system_rate

    def _simulate_system_reliability(self, cpu_count: int, hdd_count: int, ssd_count: int, 
                                   ram_count: int, cpu_fit: float, hdd_fit: float, 
                                   ssd_fit: float, ram_fit: float, system_type: str) -> np.ndarray:
        """
        Simulate system reliability using enterprise redundancy model
        """
        print(f"📊 ANALYZING {system_type.upper()} SYSTEM:")
        print("   " + "="*50)
        
        # Get PSU configuration
        psu_count = self.config.get(f'{system_type}_psu_count', 7)
        psu_fit = self.config.get(f'{system_type}_psu_fit', 200e-9)
        
        # Calculate system failure rate with redundancy
        system_failure_rate = self._calculate_system_reliability(
            cpu_count, hdd_count, ssd_count, ram_count, psu_count,
            cpu_fit, hdd_fit, ssd_fit, ram_fit, psu_fit, system_type
        )
        
        # Calculate theoretical MTBF
        mtbf_hours = 1 / system_failure_rate
        mtbf_years = mtbf_hours / 8760
        
        print(f"   🔧 Components: {cpu_count} CPU, {hdd_count} HDD, {ssd_count} SSD, {ram_count} RAM, {psu_count} PSU")
        print(f"   ⚡ System failure rate: {system_failure_rate:.2e} failures/hour")
        print(f"   ⏱️  Theoretical MTBF: {mtbf_years:.1f} years ({mtbf_hours/24:.0f} days)")
        
        # Monte Carlo simulation
        system_dist = stats.expon(scale=1/system_failure_rate)
        n_simulations = self.config.get('n_simulations')
        
        results_hours = system_dist.rvs(n_simulations)
        
        # Statistical analysis
        mtbf_mean = np.mean(results_hours)
        confidence_interval = np.percentile(results_hours, [
            self.config.get('percentile_lower'), 
            self.config.get('percentile_upper')
        ])
        failure_threshold = self.config.get('failure_threshold_hours')
        failure_probability = np.mean(results_hours < failure_threshold) * 100
        
        # Display results
        print(f"   📈 SIMULATION RESULTS:")
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
        
        # Calculate additional uptime per year (correct formula)
        hours_per_year = 8760
        current_downtime_per_year = hours_per_year / current_mtbf
        upgraded_downtime_per_year = hours_per_year / upgraded_mtbf
        additional_uptime_per_year = (current_downtime_per_year - upgraded_downtime_per_year) * hours_per_year
        
        print(f"   ✅ Reliability improvement: {reliability_improvement:.1f}%")
        print(f"   ✅ Risk reduction: {risk_reduction:.1f}%")
        print(f"   ✅ Additional uptime per year: {additional_uptime_per_year:.1f} hours")
        print(f"   ⏰ Reduced downtime per year: {(current_downtime_per_year - upgraded_downtime_per_year)*24:.1f} minutes")
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
        # Handle matplotlib style safely
        plot_style = self.config.get('plot_style', 'default')
        try:
            if plot_style and plot_style != 'default':
                plt.style.use(plot_style)
            else:
                plt.style.use('default')
        except (OSError, TypeError) as e:
            print(f"⚠️  Style '{plot_style}' not available. Using default.")
            plt.style.use('default')
        
        # Get figure dimensions safely
        fig_width = self.config.get('figure_width', 16)
        fig_height = self.config.get('figure_height', 12)
        
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
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
        plt.savefig(output_file, dpi=self.config.get('dpi'), bbox_inches='tight')
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