import React from 'react';
import AnomalyStatus from './AnomalyStatus';
import StatisticsGrid from './StatisticsGrid';
import ClassDistribution from './ClassDistribution';
import LogTypeDistribution from './LogTypeDistribution';
import ResultDetails from './ResultDetails';
import DetailedResults from './DetailedResults';
import ModelInfo from './ModelInfo';

export default function ResultsDisplay({ results }) {
  if (!results) return null;
  
  return (
    <div className="space-y-4 max-h-[600px] overflow-y-auto">
      <AnomalyStatus 
        anomalyDetected={results.anomaly_detected}
        confidence={results.confidence}
        prediction={results.primary_prediction}
        predictionClassId={results.primary_prediction_class_id}
      />
      <StatisticsGrid 
        statistics={results.statistics}
        processingTime={results.processing_time_seconds}
      />
      {results.statistics?.log_type_distribution && (
        <LogTypeDistribution
          logTypeDistribution={results.statistics.log_type_distribution}
        />
      )}
      {results.statistics?.class_distribution && (
        <ClassDistribution
          classDistribution={results.statistics.class_distribution}
          totalLines={results.statistics.total_lines}
        />
      )}
      <ResultDetails
        predictedSource={results.predicted_source}
        template={results.template}
        embeddingDims={results.embedding_dims}
        modelInfo={results.model_info}
      />
      <DetailedResults detailedResults={results.detailed_results} />
      <ModelInfo modelInfo={results.model_info} />
    </div>
  );
}