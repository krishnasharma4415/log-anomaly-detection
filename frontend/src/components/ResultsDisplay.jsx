import React from 'react';
import AnomalyStatus from './AnomalyStatus';
import StatisticsGrid from './StatisticsGrid';
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
      />
      <StatisticsGrid 
        statistics={results.statistics}
        processingTime={results.processing_time}
      />
      <ResultDetails
        predictedSource={results.predicted_source}
        template={results.template}
        embeddingDims={results.embedding_dims}
        modelUsed={results.model_used}
      />
      <DetailedResults detailedResults={results.detailed_results} />
      <ModelInfo modelUsed={results.model_used} />
    </div>
  );
}