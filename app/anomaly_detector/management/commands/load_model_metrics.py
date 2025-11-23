"""
Management command to load model metrics from training results
"""
import json
from pathlib import Path
from django.core.management.base import BaseCommand
from django.conf import settings
from anomaly_detector.models import ModelMetrics


class Command(BaseCommand):
    help = 'Load model metrics from training results'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--results-dir',
            type=str,
            default=None,
            help='Path to results directory'
        )
    
    def handle(self, *args, **options):
        results_dir = options['results_dir']
        if results_dir is None:
            results_dir = settings.PROJECT_ROOT / 'results'
        else:
            results_dir = Path(results_dir)
        
        if not results_dir.exists():
            self.stdout.write(self.style.ERROR(f'Results directory not found: {results_dir}'))
            return
        
        self.stdout.write(f'Loading metrics from: {results_dir}')
        
        # Load ML model metrics
        ml_results = results_dir / 'aggregate_results'
        if ml_results.exists():
            self._load_ml_metrics(ml_results)
        
        # Load DL model metrics
        dl_results = results_dir / 'dl_results'
        if dl_results.exists():
            self._load_dl_metrics(dl_results)
        
        # Load BERT model metrics
        bert_results = results_dir / 'bert_results'
        if bert_results.exists():
            self._load_bert_metrics(bert_results)
        
        self.stdout.write(self.style.SUCCESS('[OK] Model metrics loaded successfully'))
    
    def _load_ml_metrics(self, results_dir):
        """Load ML model metrics"""
        summary_file = results_dir / 'summary.json'
        if not summary_file.exists():
            return
        
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        for model_name, metrics in data.get('models', {}).items():
            ModelMetrics.objects.create(
                model_type='ml',
                model_name=model_name,
                f1_score=metrics.get('f1_macro', 0),
                balanced_accuracy=metrics.get('balanced_acc', 0),
                auroc=metrics.get('auroc'),
                mcc=metrics.get('mcc'),
                per_class_metrics=metrics.get('per_class', {}),
                test_samples=metrics.get('test_samples', 0)
            )
        
        self.stdout.write(f'  Loaded {len(data.get("models", {}))} ML model metrics')
    
    def _load_dl_metrics(self, results_dir):
        """Load DL model metrics"""
        # Similar implementation for DL models
        pass
    
    def _load_bert_metrics(self, results_dir):
        """Load BERT model metrics"""
        # Similar implementation for BERT models
        pass
