import React from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import FeedbackAnalysisGrid from '../components/FeedbackAnalysisGrid';

const Feedback = () => {
    const location = useLocation();
    
    // Obtener datos del state de navegación
    const { 
        jobId, 
        exerciseName, 
        fileName, 
        uploadedAt,
        // Mantener compatibilidad con versión anterior (si viene de la vieja lógica)
        originalFile 
    } = location.state || {};

    console.log('🎯 Feedback page loaded with state:', location.state);

    // Determinar nombre del ejercicio para mostrar
    const exerciseDisplayName = exerciseName || "Ejercicio";

    // Verificar que tenemos lo mínimo necesario
    if (!jobId && !originalFile) {
        return (
            <div className='page-container'>
                <Navbar />
                <div className='main-content' style={{ textAlign: 'center', padding: '40px' }}>
                    <h2 style={{ color: 'var(--font-color)' }}>No hay análisis para mostrar</h2>
                    <p style={{ color: 'var(--font-color)' }}>
                        No se encontró un job_id válido. Por favor, regresa y sube un video primero.
                    </p>
                    <div style={{ marginTop: '20px' }}>
                        <button 
                            className="feedback-back-button" 
                            onClick={() => window.history.back()}
                            style={{
                                padding: '12px 24px',
                                backgroundColor: 'var(--secondary-color)',
                                color: 'white',
                                border: 'none',
                                borderRadius: '8px',
                                cursor: 'pointer'
                            }}
                        >
                            Volver
                        </button>
                    </div>
                </div>
                <Footer />
            </div>
        );
    }

    return (
        <div className='page-container'>
            <Navbar />
            <div className='main-content'>
                <FeedbackAnalysisGrid 
                    exerciseName={exerciseDisplayName}
                    originalFile={originalFile}
                    fileName={fileName}
                    jobId={jobId}
                    uploadedAt={uploadedAt}
                />
            </div>
            <Footer />
        </div>
    );
};

export default Feedback;