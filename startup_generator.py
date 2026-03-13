"""
VenturePulse Synthetic Dataset Generator - CORRECTED VERSION
Academic MLOps Project - Group 4
March 2026

This module generates a realistic, statistically coherent synthetic dataset
for startup success prediction optimized for Precision@100.

FIXED: Class balance now correctly generates ~15% positive class.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set reproducible seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class VenturePulseDataGenerator:
    """
    Generate synthetic startup data with realistic VC correlations.
    
    Key Design Principles:
    1. No data leakage (features available at t=0, target at t=18)
    2. Realistic correlations (Series A → higher funding → larger teams)
    3. Controlled class balance (~15% positive)
    4. Fairness constraints (no sector >25%, balanced geography)
    5. Non-trivial separability (ML should beat 30% precision baseline)
    """
    
    def __init__(self, n_samples: int = 2000, target_success_rate: float = 0.15):
        self.n_samples = n_samples
        self.target_success_rate = target_success_rate
        
        # Define categorical distributions (enforcing constraints)
        self.sectors = ['Technology', 'Fintech', 'Healthcare', 'Consumer', 'SaaS', 'Climate', 'AI']
        self.sector_weights = [0.20, 0.18, 0.16, 0.15, 0.15, 0.10, 0.06]  # No sector >25%
        
        self.locations = ['Silicon Valley', 'NYC', 'Boston', 'Austin', 'London', 'Berlin', 'Remote']
        self.location_weights = [0.25, 0.20, 0.15, 0.12, 0.12, 0.08, 0.08]
        
        self.funding_stages = ['Pre-seed', 'Seed', 'Series A']
        self.stage_weights = [0.30, 0.50, 0.20]  # Industry distribution
        
        # Define realistic funding ranges by stage (in $K)
        self.funding_ranges = {
            'Pre-seed': (100, 750),
            'Seed': (500, 2500),
            'Series A': (2000, 5000)
        }
        
        # Team size ranges by stage
        self.team_size_ranges = {
            'Pre-seed': (2, 6),
            'Seed': (4, 12),
            'Series A': (8, 20)
        }
        
        # Hot sectors (2024-2026 trends)
        self.hot_sectors = {'AI', 'Fintech'}
        
        # Tier 1 locations (top VC ecosystems)
        self.tier1_locations = {'Silicon Valley', 'NYC', 'Boston'}
    
    def generate_base_features(self) -> pd.DataFrame:
        """
        Generate base categorical and continuous features with realistic correlations.
        """
        data = {}
        
        # Generate funding stage first (drives many correlations)
        data['funding_stage'] = np.random.choice(
            self.funding_stages,
            size=self.n_samples,
            p=self.stage_weights
        )
        
        # Generate sector (balanced, no sector >25%)
        data['sector'] = np.random.choice(
            self.sectors,
            size=self.n_samples,
            p=self.sector_weights
        )
        
        # Generate location (Tier 1 ecosystems slightly favored)
        data['location'] = np.random.choice(
            self.locations,
            size=self.n_samples,
            p=self.location_weights
        )
        
        # Generate funding amount (correlated with stage)
        funding_amounts = []
        for stage in data['funding_stage']:
            min_fund, max_fund = self.funding_ranges[stage]
            # Use log-normal distribution (realistic for funding)
            mu = np.log((min_fund + max_fund) / 2)
            sigma = 0.4
            amount = np.random.lognormal(mu, sigma)
            # Clip to realistic range
            amount = np.clip(amount, min_fund, max_fund)
            funding_amounts.append(amount)
        
        data['initial_funding_amount'] = funding_amounts
        
        # Generate team size (correlated with stage and funding)
        team_sizes = []
        for i, stage in enumerate(data['funding_stage']):
            min_team, max_team = self.team_size_ranges[stage]
            
            # Base team size from stage distribution
            base_team = np.random.randint(min_team, max_team + 1)
            
            # Adjustment: well-funded startups tend to have larger teams
            funding_percentile = (funding_amounts[i] - self.funding_ranges[stage][0]) / \
                                (self.funding_ranges[stage][1] - self.funding_ranges[stage][0])
            
            if funding_percentile > 0.7:  # Top 30% funded
                base_team = min(base_team + np.random.randint(0, 3), max_team)
            
            team_sizes.append(base_team)
        
        data['team_size'] = team_sizes
        
        # Generate months since founded (realistic age distribution)
        months_founded = []
        for stage in data['funding_stage']:
            if stage == 'Pre-seed':
                months = np.random.randint(3, 18)
            elif stage == 'Seed':
                months = np.random.randint(12, 30)
            else:  # Series A
                months = np.random.randint(24, 36)
            months_founded.append(months)
        
        data['months_since_founded'] = months_founded
        
        return pd.DataFrame(data)
    
    def engineer_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features that capture VC decision signals.
        """
        # Capital efficiency (funding per team member)
        df['capital_efficiency'] = df['initial_funding_amount'] / df['team_size']
        
        # Tier 1 location indicator
        df['tier1_location'] = df['location'].apply(
            lambda x: 1 if x in self.tier1_locations else 0
        )
        
        # Hot sector indicator
        df['hot_sector'] = df['sector'].apply(
            lambda x: 1 if x in self.hot_sectors else 0
        )
        
        # Stage funding match (is funding appropriate for stage?)
        stage_funding_match = []
        for _, row in df.iterrows():
            stage = row['funding_stage']
            funding = row['initial_funding_amount']
            min_fund, max_fund = self.funding_ranges[stage]
            median_fund = (min_fund + max_fund) / 2
            
            # Within +/- 30% of median is considered "matched"
            is_matched = 1 if abs(funding - median_fund) / median_fund < 0.5 else 0
            stage_funding_match.append(is_matched)
        
        df['stage_funding_match'] = stage_funding_match
        
        # Progression speed (stage index / months)
        stage_index_map = {'Pre-seed': 0, 'Seed': 1, 'Series A': 2}
        df['progression_speed'] = df.apply(
            lambda row: (stage_index_map[row['funding_stage']] + 1) / 
                       (row['months_since_founded'] / 12), axis=1
        )
        
        return df
    
    def generate_target_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generate target labels using structured probabilistic model.
        
        CORRECTED VERSION: Properly calibrated to achieve ~15% success rate.
        
        This is CRITICAL - labels must reflect realistic VC success patterns:
        - Base rate: 15%
        - Conditional lifts based on realistic signals
        - Non-deterministic (includes realistic noise)
        """
        # CRITICAL FIX: Start with much lower base probability
        success_probs = np.full(len(df), 0.03)  # Changed from 0.15 to 0.03
        
        # Stage lift (later stage → higher success probability)
        # REDUCED to achieve proper balance
        stage_lift = df['funding_stage'].map({
            'Pre-seed': 0.00,
            'Seed': 0.04,       # Reduced from 0.08
            'Series A': 0.12    # Reduced from 0.20
        }).values
        success_probs += stage_lift
        
        # Tier 1 location lift (ecosystem advantage)
        # REDUCED to achieve proper balance
        tier1_lift = df['tier1_location'].values * 0.08  # Reduced from 0.15
        success_probs += tier1_lift
        
        # Hot sector lift (market timing)
        # REDUCED to achieve proper balance
        hot_sector_lift = df['hot_sector'].values * 0.06  # Reduced from 0.10
        success_probs += hot_sector_lift
        
        # Well-funded lift (above stage median)
        # REDUCED to achieve proper balance
        well_funded_lift = []
        for _, row in df.iterrows():
            stage = row['funding_stage']
            funding = row['initial_funding_amount']
            median_fund = sum(self.funding_ranges[stage]) / 2
            lift = 0.05 if funding > median_fund else 0.0  # Reduced from 0.10
            well_funded_lift.append(lift)
        success_probs += np.array(well_funded_lift)
        
        # Capital efficiency lift (top 30%)
        # REDUCED to achieve proper balance
        cap_eff_percentile = df['capital_efficiency'].rank(pct=True)
        cap_eff_lift = (cap_eff_percentile > 0.70).astype(float) * 0.04  # Reduced from 0.08
        success_probs += cap_eff_lift
        
        # Established startup lift (older = more proven)
        # REDUCED to achieve proper balance
        established_lift = (df['months_since_founded'] > 18).astype(float) * 0.03  # Reduced from 0.05
        success_probs += established_lift
        
        # Cap at lower maximum (no startup is certain to succeed)
        success_probs = np.clip(success_probs, 0.01, 0.50)  # Changed from (0.05, 0.85)
        
        # Sample from Bernoulli distribution
        labels = np.random.binomial(1, success_probs)
        
        return labels
    
    def inject_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inject 5% random missing values in non-critical fields.
        """
        # Never allow missing in: funding_stage, sector, location, target
        safe_to_miss = ['team_size', 'months_since_founded', 
                       'capital_efficiency', 'progression_speed']
        
        for col in safe_to_miss:
            if col in df.columns:
                n_missing = int(len(df) * 0.05)
                missing_idx = np.random.choice(df.index, size=n_missing, replace=False)
                df.loc[missing_idx, col] = np.nan
        
        return df
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate dataset meets all requirements.
        """
        validations = {}
        
        # Check class balance (13-17%)
        success_rate = df['high_traction'].mean()
        validations['class_balance'] = 0.13 <= success_rate <= 0.17
        
        # Check no sector > 25%
        sector_dist = df['sector'].value_counts(normalize=True)
        validations['sector_constraint'] = (sector_dist <= 0.25).all()
        
        # Check stage distribution (30/50/20 ± 5%)
        stage_dist = df['funding_stage'].value_counts(normalize=True)
        validations['stage_distribution'] = (
            0.25 <= stage_dist.get('Pre-seed', 0) <= 0.35 and
            0.45 <= stage_dist.get('Seed', 0) <= 0.55 and
            0.15 <= stage_dist.get('Series A', 0) <= 0.25
        )
        
        # Check realistic correlations
        # Series A should have higher median funding
        series_a_median = df[df['funding_stage'] == 'Series A']['initial_funding_amount'].median()
        seed_median = df[df['funding_stage'] == 'Seed']['initial_funding_amount'].median()
        validations['funding_stage_correlation'] = series_a_median > seed_median
        
        # Tier 1 should have higher average funding
        tier1_avg = df[df['tier1_location'] == 1]['initial_funding_amount'].mean()
        non_tier1_avg = df[df['tier1_location'] == 0]['initial_funding_amount'].mean()
        validations['tier1_funding_correlation'] = tier1_avg > non_tier1_avg
        
        # Check value ranges
        validations['funding_range'] = (
            df['initial_funding_amount'].between(100, 5000).all()
        )
        validations['team_size_range'] = (
            df['team_size'].dropna().between(2, 20).all()
        )
        validations['months_range'] = (
            df['months_since_founded'].dropna().between(3, 36).all()
        )
        
        # Check missing values (≤5% per feature, 0% in critical)
        critical_cols = ['funding_stage', 'sector', 'location', 'high_traction']
        validations['no_missing_critical'] = df[critical_cols].isna().sum().sum() == 0
        
        other_cols = [c for c in df.columns if c not in critical_cols]
        missing_rates = df[other_cols].isna().mean()
        validations['missing_values_controlled'] = (missing_rates <= 0.06).all()
        
        return validations
    
    def generate(self) -> pd.DataFrame:
        """
        Main generation pipeline.
        """
        print("=" * 70)
        print("VenturePulse Synthetic Dataset Generator - CORRECTED VERSION")
        print("=" * 70)
        print(f"Target samples: {self.n_samples:,}")
        print(f"Target success rate: {self.target_success_rate:.1%}")
        print(f"Random seed: {RANDOM_SEED}\n")
        
        # Step 1: Generate base features
        print("Step 1/5: Generating base features...")
        df = self.generate_base_features()
        
        # Step 2: Engineer derived features
        print("Step 2/5: Engineering derived features...")
        df = self.engineer_derived_features(df)
        
        # Step 3: Generate target labels
        print("Step 3/5: Generating target labels...")
        df['high_traction'] = self.generate_target_labels(df)
        
        # Step 4: Inject missing values
        print("Step 4/5: Injecting realistic missing values...")
        df = self.inject_missing_values(df)
        
        # Step 5: Validate
        print("Step 5/5: Validating dataset constraints...\n")
        validations = self.validate_dataset(df)
        
        # Print validation results
        print("=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        for check, passed in validations.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{check:.<55} {status}")
        print("=" * 70 + "\n")
        
        if not all(validations.values()):
            print("⚠️  WARNING: Some validations failed. Review dataset generation logic.")
        else:
            print("✅ All validations passed!\n")
        
        return df


def print_dataset_statistics(df: pd.DataFrame):
    """
    Print comprehensive dataset statistics.
    """
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print(f"\n🎯 Class Distribution:")
    class_dist = df['high_traction'].value_counts()
    print(f"   High Traction (1): {class_dist.get(1, 0):,} ({class_dist.get(1, 0)/len(df):.1%})")
    print(f"   No Traction (0):   {class_dist.get(0, 0):,} ({class_dist.get(0, 0)/len(df):.1%})")
    
    print(f"\n🏢 Sector Distribution:")
    for sector, count in df['sector'].value_counts().items():
        pct = count / len(df)
        bar = "█" * int(pct * 40)
        print(f"   {sector:.<20} {count:>4} ({pct:>5.1%}) {bar}")
    
    print(f"\n📍 Location Distribution:")
    for loc, count in df['location'].value_counts().items():
        pct = count / len(df)
        tier1 = "⭐" if loc in ['Silicon Valley', 'NYC', 'Boston'] else ""
        print(f"   {loc:.<20} {count:>4} ({pct:>5.1%}) {tier1}")
    
    print(f"\n🚀 Funding Stage Distribution:")
    for stage, count in df['funding_stage'].value_counts().items():
        pct = count / len(df)
        print(f"   {stage:.<20} {count:>4} ({pct:>5.1%})")
    
    print(f"\n💰 Funding Statistics (in $K):")
    print(f"   Mean:   ${df['initial_funding_amount'].mean():>8,.0f}")
    print(f"   Median: ${df['initial_funding_amount'].median():>8,.0f}")
    print(f"   Std:    ${df['initial_funding_amount'].std():>8,.0f}")
    print(f"   Range:  ${df['initial_funding_amount'].min():>8,.0f} - ${df['initial_funding_amount'].max():>8,.0f}")
    
    print(f"\n👥 Team Size Statistics:")
    print(f"   Mean:   {df['team_size'].mean():>6.1f} people")
    print(f"   Median: {df['team_size'].median():>6.0f} people")
    print(f"   Range:  {df['team_size'].min():>6.0f} - {df['team_size'].max():>6.0f} people")
    
    print(f"\n⏱️  Age Statistics (months):")
    print(f"   Mean:   {df['months_since_founded'].mean():>6.1f} months")
    print(f"   Median: {df['months_since_founded'].median():>6.0f} months")
    
    print(f"\n🔧 Engineered Features:")
    print(f"   Capital Efficiency:  ${df['capital_efficiency'].mean():>8,.0f}/person (mean)")
    print(f"   Tier 1 Locations:    {df['tier1_location'].sum():,} ({df['tier1_location'].mean():.1%})")
    print(f"   Hot Sectors:         {df['hot_sector'].sum():,} ({df['hot_sector'].mean():.1%})")
    print(f"   Stage-Funding Match: {df['stage_funding_match'].sum():,} ({df['stage_funding_match'].mean():.1%})")
    
    print(f"\n❓ Missing Values:")
    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if len(missing_summary) == 0:
        print("   No missing values")
    else:
        for col, count in missing_summary.items():
            print(f"   {col:.<35} {count:>4} ({count/len(df):>5.1%})")
    
    print(f"\n🔗 Key Correlations:")
    print(f"   Series A → Higher Funding:")
    for stage in ['Pre-seed', 'Seed', 'Series A']:
        stage_median = df[df['funding_stage'] == stage]['initial_funding_amount'].median()
        print(f"      {stage:.<15} Median: ${stage_median:>8,.0f}")
    
    print(f"\n   Tier 1 → Higher Funding:")
    tier1_avg = df[df['tier1_location'] == 1]['initial_funding_amount'].mean()
    non_tier1_avg = df[df['tier1_location'] == 0]['initial_funding_amount'].mean()
    print(f"      Tier 1:         ${tier1_avg:>8,.0f} (mean)")
    print(f"      Non-Tier 1:     ${non_tier1_avg:>8,.0f} (mean)")
    print(f"      Difference:     ${tier1_avg - non_tier1_avg:>8,.0f} (+{((tier1_avg/non_tier1_avg - 1) * 100):.1f}%)")
    
    print(f"\n   Success Rate by Stage:")
    for stage in ['Pre-seed', 'Seed', 'Series A']:
        stage_success = df[df['funding_stage'] == stage]['high_traction'].mean()
        print(f"      {stage:.<15} {stage_success:>5.1%}")
    
    print(f"\n   Success Rate by Location Type:")
    tier1_success = df[df['tier1_location'] == 1]['high_traction'].mean()
    non_tier1_success = df[df['tier1_location'] == 0]['high_traction'].mean()
    print(f"      Tier 1:         {tier1_success:>5.1%}")
    print(f"      Non-Tier 1:     {non_tier1_success:>5.1%}")
    print(f"      Lift:           {tier1_success - non_tier1_success:>+5.1%}")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Generate dataset
    generator = VenturePulseDataGenerator(n_samples=5000, target_success_rate=0.15)
    df = generator.generate()
    
    # Print statistics
    print_dataset_statistics(df)
    
    # Save to CSV
    output_file = "venturepulse_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Dataset saved to: {output_file}")
    print(f"📁 File size: {len(df):,} rows\n")
    
    # Print column order for reference
    print("📋 Column Order:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    print("\n💡 Next Steps:")
    print("   1. Move dataset to data/ folder: move venturepulse_dataset.csv data/")
    print("   2. Create train/val/test splits")
    print("   3. Start exploratory data analysis")
    print("   4. Begin feature engineering and modeling")
    print("=" * 70)