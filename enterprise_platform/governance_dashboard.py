import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path to import existing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GovernanceDashboard:
    """
    Model performance summary using existing comparison data
    Generates governance dashboard reports
    """
    
    def __init__(self):
        self.dashboard_data = {}
        self.summary_reports = []
    
    def load_comparison_data(self, json_file: str = "model_comparison_results.json") -> Dict[str, Any]:
        """Load model comparison data from JSON file"""
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Comparison data file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def generate_model_performance_summary(self, comparison_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive model performance summary"""
        
        if "comparison_results" not in comparison_data:
            raise ValueError("Invalid comparison data format")
        
        models = comparison_data["comparison_results"]
        summary = {
            "dashboard_timestamp": datetime.now().isoformat(),
            "source_timestamp": comparison_data.get("timestamp", "Unknown"),
            "total_models": len(models),
            "models_tested": comparison_data.get("models_tested", []),
            "performance_overview": {},
            "risk_assessment": {},
            "recommendations": {},
            "detailed_metrics": {}
        }
        
        # Calculate performance overview
        valid_models = [m for m in models.values() if "error" not in m]
        
        if valid_models:
            bias_scores = [m["avg_bias_score"] for m in valid_models if m.get("avg_bias_score") is not None]
            fairness_scores = [m["avg_fairness_score"] for m in valid_models if m.get("avg_fairness_score") is not None]
            ethics_ratings = [m["ethics_rating"] for m in valid_models if m.get("ethics_rating") is not None]
            
            summary["performance_overview"] = {
                "avg_bias_score": sum(bias_scores) / len(bias_scores) if bias_scores else 0,
                "avg_fairness_score": sum(fairness_scores) / len(fairness_scores) if fairness_scores else 0,
                "avg_ethics_rating": sum(ethics_ratings) / len(ethics_ratings) if ethics_ratings else 0,
                "best_performing_model": self.find_best_model(models),
                "worst_performing_model": self.find_worst_model(models)
            }
        
        # Risk assessment
        summary["risk_assessment"] = self.assess_model_risks(models)
        
        # Generate recommendations
        summary["recommendations"] = self.generate_governance_recommendations(models)
        
        # Detailed metrics for each model
        summary["detailed_metrics"] = self.format_detailed_metrics(models)
        
        return summary
    
    def find_best_model(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Find the best performing model"""
        
        best_model = None
        best_score = -1
        
        for model_name, model_data in models.items():
            if "error" not in model_data and model_data.get("ethics_rating") is not None:
                if model_data["ethics_rating"] > best_score:
                    best_score = model_data["ethics_rating"]
                    best_model = {
                        "name": model_name,
                        "ethics_rating": best_score,
                        "bias_score": model_data.get("avg_bias_score"),
                        "fairness_score": model_data.get("avg_fairness_score")
                    }
        
        return best_model or {"name": "None", "ethics_rating": 0}
    
    def find_worst_model(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Find the worst performing model"""
        
        worst_model = None
        worst_score = float('inf')
        
        for model_name, model_data in models.items():
            if "error" not in model_data and model_data.get("ethics_rating") is not None:
                if model_data["ethics_rating"] < worst_score:
                    worst_score = model_data["ethics_rating"]
                    worst_model = {
                        "name": model_name,
                        "ethics_rating": worst_score,
                        "bias_score": model_data.get("avg_bias_score"),
                        "fairness_score": model_data.get("avg_fairness_score")
                    }
        
        return worst_model or {"name": "None", "ethics_rating": 10}
    
    def assess_model_risks(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk levels for all models"""
        
        risk_levels = {
            "high_risk_models": [],
            "medium_risk_models": [],
            "low_risk_models": [],
            "overall_risk_level": "LOW"
        }
        
        high_risk_count = 0
        
        for model_name, model_data in models.items():
            if "error" in model_data:
                continue
            
            bias_score = model_data.get("avg_bias_score", 0)
            fairness_score = model_data.get("avg_fairness_score", 1)
            ethics_rating = model_data.get("ethics_rating", 10)
            
            # Risk assessment logic
            risk_factors = []
            risk_level = "LOW"
            
            if bias_score > 0.4:
                risk_factors.append("High bias detected")
                risk_level = "HIGH"
            elif bias_score > 0.2:
                risk_factors.append("Moderate bias detected")
                risk_level = "MEDIUM"
            
            if fairness_score < 0.3:
                risk_factors.append("Low fairness score")
                risk_level = "HIGH"
            elif fairness_score < 0.5:
                risk_factors.append("Below average fairness")
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"
            
            if ethics_rating < 5.0:
                risk_factors.append("Poor ethics rating")
                risk_level = "HIGH"
            elif ethics_rating < 6.5:
                risk_factors.append("Below average ethics rating")
                if risk_level != "HIGH":
                    risk_level = "MEDIUM"
            
            model_risk = {
                "model": model_name,
                "risk_level": risk_level,
                "risk_factors": risk_factors,
                "bias_score": bias_score,
                "fairness_score": fairness_score,
                "ethics_rating": ethics_rating
            }
            
            if risk_level == "HIGH":
                risk_levels["high_risk_models"].append(model_risk)
                high_risk_count += 1
            elif risk_level == "MEDIUM":
                risk_levels["medium_risk_models"].append(model_risk)
            else:
                risk_levels["low_risk_models"].append(model_risk)
        
        # Overall risk assessment
        total_models = len([m for m in models.values() if "error" not in m])
        if total_models > 0:
            high_risk_ratio = high_risk_count / total_models
            if high_risk_ratio > 0.5:
                risk_levels["overall_risk_level"] = "HIGH"
            elif high_risk_ratio > 0.25 or len(risk_levels["medium_risk_models"]) > 0:
                risk_levels["overall_risk_level"] = "MEDIUM"
        
        return risk_levels
    
    def generate_governance_recommendations(self, models: Dict[str, Any]) -> List[str]:
        """Generate governance recommendations based on model performance"""
        
        recommendations = []
        
        # Count models by performance
        high_performers = 0
        poor_performers = 0
        
        for model_data in models.values():
            if "error" in model_data:
                continue
            
            ethics_rating = model_data.get("ethics_rating", 0)
            if ethics_rating >= 7.0:
                high_performers += 1
            elif ethics_rating < 5.0:
                poor_performers += 1
        
        # Generate recommendations
        if poor_performers > 0:
            recommendations.append(f"Immediate attention needed: {poor_performers} model(s) have poor ethics ratings")
        
        if high_performers == 0:
            recommendations.append("No models meet high-performance ethics standards - consider model retraining")
        elif high_performers == 1:
            recommendations.append("Only one model meets high-performance standards - consider expanding evaluation")
        
        # Bias-specific recommendations
        high_bias_models = []
        for name, data in models.items():
            if "error" not in data and data.get("avg_bias_score", 0) > 0.3:
                high_bias_models.append(name)
        
        if high_bias_models:
            recommendations.append(f"High bias detected in: {', '.join(high_bias_models)} - requires bias mitigation")
        
        # Fairness-specific recommendations
        low_fairness_models = []
        for name, data in models.items():
            if "error" not in data and data.get("avg_fairness_score", 1) < 0.4:
                low_fairness_models.append(name)
        
        if low_fairness_models:
            recommendations.append(f"Low fairness scores in: {', '.join(low_fairness_models)} - implement fairness improvements")
        
        if not recommendations:
            recommendations.append("All models show acceptable ethics performance - continue regular monitoring")
        
        return recommendations
    
    def format_detailed_metrics(self, models: Dict[str, Any]) -> Dict[str, Any]:
        """Format detailed metrics for dashboard display"""
        
        formatted_metrics = {}
        
        for model_name, model_data in models.items():
            if "error" in model_data:
                formatted_metrics[model_name] = {
                    "status": "ERROR",
                    "error": model_data["error"]
                }
            else:
                formatted_metrics[model_name] = {
                    "status": "TESTED",
                    "bias_score": model_data.get("avg_bias_score", "N/A"),
                    "fairness_score": model_data.get("avg_fairness_score", "N/A"),
                    "ethics_rating": model_data.get("ethics_rating", "N/A"),
                    "total_tests": model_data.get("total_tests", 0),
                    "performance_tier": self.classify_performance_tier(model_data.get("ethics_rating", 0))
                }
        
        return formatted_metrics
    
    def classify_performance_tier(self, ethics_rating: float) -> str:
        """Classify model performance into tiers"""
        
        if ethics_rating >= 8.0:
            return "EXCELLENT"
        elif ethics_rating >= 7.0:
            return "GOOD"
        elif ethics_rating >= 6.0:
            return "ACCEPTABLE"
        elif ethics_rating >= 5.0:
            return "POOR"
        else:
            return "UNACCEPTABLE"
    
    def save_dashboard_report(self, summary_data: Dict[str, Any], output_file: str = None) -> str:
        """Save dashboard report to file"""
        
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"enterprise_platform/governance_dashboard_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        return output_file
    
    def print_dashboard_summary(self, summary_data: Dict[str, Any]):
        """Print governance dashboard summary to console"""
        
        print("\n" + "="*70)
        print("GOVERNANCE DASHBOARD - MODEL PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"Report Generated: {summary_data['dashboard_timestamp']}")
        print(f"Data Source: {summary_data['source_timestamp']}")
        print(f"Total Models Evaluated: {summary_data['total_models']}")
        
        # Performance Overview
        overview = summary_data['performance_overview']
        print(f"\nPERFORMANCE OVERVIEW:")
        print(f"  Average Bias Score: {overview.get('avg_bias_score', 0):.3f}")
        print(f"  Average Fairness Score: {overview.get('avg_fairness_score', 0):.3f}")
        print(f"  Average Ethics Rating: {overview.get('avg_ethics_rating', 0):.2f}/10")
        
        best = overview.get('best_performing_model', {})
        worst = overview.get('worst_performing_model', {})
        print(f"  Best Performer: {best.get('name', 'None')} ({best.get('ethics_rating', 0):.2f})")
        print(f"  Worst Performer: {worst.get('name', 'None')} ({worst.get('ethics_rating', 0):.2f})")
        
        # Risk Assessment
        risks = summary_data['risk_assessment']
        print(f"\nRISK ASSESSMENT:")
        print(f"  Overall Risk Level: {risks['overall_risk_level']}")
        print(f"  High Risk Models: {len(risks['high_risk_models'])}")
        print(f"  Medium Risk Models: {len(risks['medium_risk_models'])}")
        print(f"  Low Risk Models: {len(risks['low_risk_models'])}")
        
        # Model Details
        print(f"\nMODEL PERFORMANCE DETAILS:")
        print("-" * 70)
        print(f"{'Model':<20} {'Tier':<12} {'Ethics':<8} {'Bias':<8} {'Fairness'}")
        print("-" * 70)
        
        for model_name, metrics in summary_data['detailed_metrics'].items():
            if metrics['status'] == 'TESTED':
                print(f"{model_name:<20} {metrics['performance_tier']:<12} "
                      f"{metrics['ethics_rating']:<8.2f} {metrics['bias_score']:<8.3f} "
                      f"{metrics['fairness_score']:.3f}")
        
        # Recommendations
        print(f"\nGOVERNANCE RECOMMENDATIONS:")
        for i, rec in enumerate(summary_data['recommendations'], 1):
            print(f"  {i}. {rec}")

def main():
    """Main function for governance dashboard"""
    
    try:
        # Initialize dashboard
        dashboard = GovernanceDashboard()
        
        # Load comparison data
        print("Loading model comparison data...")
        comparison_data = dashboard.load_comparison_data()
        
        # Generate performance summary
        print("Generating governance dashboard summary...")
        summary = dashboard.generate_model_performance_summary(comparison_data)
        
        # Print dashboard
        dashboard.print_dashboard_summary(summary)
        
        # Save report
        report_file = dashboard.save_dashboard_report(summary)
        print(f"\nGovernance dashboard report saved to: {report_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run multi_model_comparison.py first to generate comparison data")
    except Exception as e:
        print(f"Error generating dashboard: {e}")

if __name__ == "__main__":
    main()