import React from 'react';
import './ResultsViewer.css';

const ResultsViewer = ({ originalImage, resultImage, isLoading, onReset }) => {
  return (
    <div className="results-container glass-panel">
      <div className="results-header">
        <h2>Analysis Results</h2>
        <button className="btn btn-secondary" onClick={onReset}>
          Analyze Another Image
        </button>
      </div>
      
      <div className="images-grid">
        <div className="image-card">
          <h3>Original Image</h3>
          <div className="image-wrapper">
            <img src={originalImage} alt="Original" />
          </div>
        </div>
        
        <div className="image-card">
          <h3>Detected Classes</h3>
          <div className="image-wrapper">
            {isLoading ? (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>Running YOLOv8 inference...</p>
              </div>
            ) : resultImage ? (
              <img src={resultImage} alt="Result" className="result-img" />
            ) : (
              <div className="error-state">
                <p>Failed to generate results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsViewer;
