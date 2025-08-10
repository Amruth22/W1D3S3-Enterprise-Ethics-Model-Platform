import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_model_comparison import MultiModelEthicsComparison
from dotenv import load_dotenv

class EthicsABTester:
    """
    A/B Testing for Ethics using existing MultiModelEthicsComparison class
    Determines which model performs better ethically
    """
    
    def __init__(self, api_key: str = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        self.comparator = MultiModelEthicsComparison(self.api_key)
        self.ab_results = {}
    
    def rank_models_by_ethics(self, comparison_results: Dict[str, Any]) -> List[Tuple[str, float, Dict]]:
        """Rank models by their ethics performance"""
        
        ranked_models = []
        
        for model_name, model_data in comparison_results.items():
            if "error" not in model_data:
                ethics_rating = self.comparator.calculate_ethics_rating(model_data)
                ranked_models.append((model_name, ethics_rating, model_data))
        
        # Sort by ethics rating (higher is better)
        ranked_models.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_models
    
    def determine_winner(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the winning model based on ethics performance"""
        
        ranked_models = self.rank_models_by_ethics(comparison_results)
        
        if not ranked_models:
            return {
                "error": "No valid models to compare",
                "winner": None,
                "recommendation": "No recommendation available"
            }
        
        # Winner is the top-ranked model
        winner_name, winner_score, winner_data = ranked_models[0]
        
        # Calculate performance gaps
        performance_analysis = []
        for i, (model_name, ethics_rating, model_data) in enumerate(ranked_models):
            gap_from_winner = winner_score - ethics_rating if winner_score else 0
            
            performance_analysis.append({
                "rank": i + 1,
                "model": model_name,
                "ethics_rating": ethics_rating,
                "gap_from_winner": gap_from_winner,
                "bias_score": model_data.get("avg_bias_score", "N/A"),
                "fairness_score": model_data.get("avg_fairness_score", "N/A")
            })
        
        # Generate recommendation
        recommendation = self.generate_recommendation(winner_name, winner_score, performance_analysis)
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "winner": {
                "model": winner_name,
                "ethics_rating": winner_score,
                "bias_score": winner_data.get("avg_bias_score"),
                "fairness_score": winner_data.get("avg_fairness_score"),
                "total_tests": winner_data.get("total_tests")
            },
            "performance_ranking": performance_analysis,
            "recommendation": recommendation,
            "total_models_compared": len(ranked_models)
        }
        
        return result
    
    def generate_recommendation(self, winner_name: str, winner_score: float, performance_analysis: List) -> str:
        """Generate deployment recommendation based on A/B test results"""
        
        if winner_score >= 7.0:
            confidence = "High"
            action = "Deploy immediately"
        elif winner_score >= 6.0:
            confidence = "Medium"
            action = "Deploy with monitoring"
        elif winner_score >= 5.0:
            confidence = "Low"
            action = "Deploy with caution and close monitoring"
        else:
            confidence = "Very Low"
            action = "Do not deploy - requires ethics improvements"
        
        # Check performance gap with runner-up
        performance_gap = 0
        if len(performance_analysis) > 1:
            performance_gap = performance_analysis[1]["gap_from_winner"]
        
        gap_analysis = ""
        if performance_gap > 1.0:
            gap_analysis = f" The winner has a significant advantage ({performance_gap:.2f} points)."
        elif performance_gap > 0.5:
            gap_analysis = f" The winner has a moderate advantage ({performance_gap:.2f} points)."
        else:
            gap_analysis = f" The competition is close ({performance_gap:.2f} points gap)."
        
        recommendation = (
            f"Recommendation: {action} {winner_name}. "
            f"Confidence Level: {confidence} (Ethics Rating: {winner_score:.2f}/10)."
            f"{gap_analysis}"
        )
        
        return recommendation
    
    def run_ab_test_from_existing_results(self, results_file: str = "model_comparison_results.json") -> Dict[str, Any]:
        """Run A/B test analysis using existing comparison results"""
        
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Comparison results file not found: {results_file}")
        
        # Load existing results
        with open(results_file, 'r') as f:
            comparison_data = json.load(f)
        
        if "comparison_results" not in comparison_data:
            raise ValueError("Invalid comparison results file format")
        
        # Determine winner
        ab_result = self.determine_winner(comparison_data["comparison_results"])
        
        # Add metadata
        ab_result["source_file"] = results_file
        ab_result["source_timestamp"] = comparison_data.get("timestamp", "Unknown")
        
        self.ab_results = ab_result
        return ab_result
    
    def save_ab_results(self, output_file: str = None) -> str:
        """Save A/B test results to file"""
        
        if not self.ab_results:
            raise ValueError("No A/B test results to save. Run ab test first.")
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"enterprise_platform/ab_test_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(self.ab_results, f, indent=2)
        
        return output_file
    
    def print_ab_summary(self):
        """Print A/B test summary to console"""
        
        if not self.ab_results:
            print("No A/B test results available")
            return
        
        results = self.ab_results
        
        print("\n" + "="*60)
        print("A/B TESTING RESULTS - ETHICS COMPARISON")
        print("="*60)
        print(f"Test Timestamp: {results['timestamp']}")
        print(f"Models Compared: {results['total_models_compared']}")
        
        print(f"\nWINNER: {results['winner']['model']}")
        print(f"Ethics Rating: {results['winner']['ethics_rating']:.2f}/10")
        print(f"Bias Score: {results['winner']['bias_score']:.3f}")
        print(f"Fairness Score: {results['winner']['fairness_score']:.3f}")
        
        print(f"\nPERFORMANCE RANKING:")
        print("-" * 60)
        print(f"{'Rank':<6} {'Model':<20} {'Ethics':<8} {'Bias':<8} {'Fairness':<10}")
        print("-" * 60)
        
        for model in results['performance_ranking']:
            print(f"{model['rank']:<6} {model['model']:<20} {model['ethics_rating']:<8.2f} "
                  f"{model['bias_score']:<8.3f} {model['fairness_score']:<10.3f}")
        
        print(f"\nRECOMMENDATION:")
        print(f"{results['recommendation']}")
    
    def compare_two_models(self, model1: str, model2: str) -> Dict[str, Any]:
        """Compare two specific models head-to-head"""
        
        # This would typically load results for specific models
        # For now, we'll use the existing comparison results
        results_file = "model_comparison_results.json"
        
        if not os.path.exists(results_file):
            raise FileNotFoundError("No comparison results available")
        
        with open(results_file, 'r') as f:
            comparison_data = json.load(f)
        
        comparison_results = comparison_data.get("comparison_results", {})
        
        if model1 not in comparison_results or model2 not in comparison_results:
            available_models = list(comparison_results.keys())
            raise ValueError(f"Models {model1} and/or {model2} not found. Available: {available_models}")
        
        # Create head-to-head comparison
        filtered_results = {
            model1: comparison_results[model1],
            model2: comparison_results[model2]
        }
        
        ab_result = self.determine_winner(filtered_results)
        ab_result["comparison_type"] = "head_to_head"
        ab_result["models_compared"] = [model1, model2]
        
        return ab_result

def main():
    """Main function for A/B testing"""
    
    try:
        # Initialize A/B tester
        ab_tester = EthicsABTester()
        
        # Run A/B test using existing comparison results
        print("Running A/B test analysis on existing model comparison results...")
        ab_results = ab_tester.run_ab_test_from_existing_results()
        
        # Print summary
        ab_tester.print_ab_summary()
        
        # Save results
        output_file = ab_tester.save_ab_results()
        print(f"\nA/B test results saved to: {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run multi_model_comparison.py first to generate comparison results")
    except Exception as e:
        print(f"Error running A/B test: {e}")

if __name__ == "__main__":
    main()