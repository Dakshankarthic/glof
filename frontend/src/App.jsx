import React, { useState } from 'react';
import './App.css';
import ImageUploader from './components/ImageUploader';
import ResultsViewer from './components/ResultsViewer';

function App() {
  const [originalImage, setOriginalImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = async (file) => {
    setIsLoading(true);
    setError(null);
    setResultImage(null);
    
    // Set preview for original image
    const objectUrl = URL.createObjectURL(file);
    setOriginalImage(objectUrl);

    // Prepare form data
    const formData = new FormData();
    formData.append('file', file);

    try {
      const API_URL = 'https://dk1112-glof-detection-api.hf.space';
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
      }

      const blob = await response.blob();
      const resultObjectUrl = URL.createObjectURL(blob);
      setResultImage(resultObjectUrl);
    } catch (err) {
      console.error("Inference Error:", err);
      setError("Failed to process image. Make sure the backend is running and the model is loaded.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setOriginalImage(null);
    setResultImage(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>GLOF Detection Explorer</h1>
        <p>Glacial Lake Outburst Flood Detection via YOLOv8</p>
      </header>
      
      <main className="app-main">
        {!originalImage ? (
          <ImageUploader onUpload={handleImageUpload} />
        ) : (
          <ResultsViewer 
            originalImage={originalImage} 
            resultImage={resultImage} 
            isLoading={isLoading} 
            onReset={handleReset} 
          />
        )}
        
        {error && (
          <div className="error-message glass-panel">
            {error}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
