import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { MdArrowBack } from 'react-icons/md';
import FeedbackChat from './FeedbackChat';
import FeedbackVideoPlayer from './FeedbackVideoPlayer';
import FeedbackRadarChart from './FeedbackRadarChart';

const FeedbackAnalysisGrid = ({ exerciseName, originalFile, fileName, jobId }) => {
    const navigate = useNavigate();
    const pollingIntervalRef = useRef(null);
    
    // Estados para el análisis real basado en la API
    const [analysisState, setAnalysisState] = useState({
        status: 'processing', // processing, completed, error
        currentStep: 'initializing',
        completedSteps: [],
        assets: {
            feedback: { ready: false, loading: true, data: null, error: null },
            video: { ready: false, loading: true, data: null, error: null },
            radar: { ready: false, loading: true, data: null, error: null },
            report: { ready: false, loading: true, data: null, error: null }
        },
        error: null
    });

    // Polling a la API para obtener el estado del job
    const pollJobStatus = async () => {
        if (!jobId) return;

        try {
            console.log(`🔄 Polling job status for ${jobId}...`);
            const response = await fetch(`http://localhost:8000/jobs/${jobId}`);
            
            if (!response.ok) {
                throw new Error(`Error ${response.status}: ${response.statusText}`);
            }

            const statusData = await response.json();
            console.log('📊 Job status:', statusData);
            
            // Actualizar estado general
            setAnalysisState(prev => ({
                ...prev,
                status: statusData.status,
                currentStep: statusData.current_step,
                completedSteps: statusData.completed_steps,
                error: statusData.error,
                assets: {
                    // Actualizar estado de cada asset
                    feedback: {
                        ...prev.assets.feedback,
                        ready: statusData.assets_ready.feedback,
                        loading: !statusData.assets_ready.feedback && statusData.status === 'processing'
                    },
                    video: {
                        ...prev.assets.video,
                        ready: statusData.assets_ready.video,
                        loading: !statusData.assets_ready.video && statusData.status === 'processing'
                    },
                    radar: {
                        ...prev.assets.radar,
                        ready: statusData.assets_ready.radar,
                        loading: !statusData.assets_ready.radar && statusData.status === 'processing'
                    },
                    report: {
                        ...prev.assets.report,
                        ready: statusData.assets_ready.report,
                        loading: !statusData.assets_ready.report && statusData.status === 'processing'
                    }
                }
            }));

            // Cargar contenido de assets cuando estén listos
            await loadAssetData(statusData.assets_ready);

            // Parar polling si el análisis está completo o hay error
            if (statusData.status === 'completed' || statusData.status === 'error') {
                console.log(`✅ Polling stopped - Status: ${statusData.status}`);
                stopPolling();
            }

        } catch (error) {
            console.error('❌ Error polling job status:', error);
            setAnalysisState(prev => ({
                ...prev,
                status: 'error',
                error: `Error de conexión: ${error.message}`
            }));
            stopPolling();
        }
    };

    // Cargar datos específicos de los assets cuando estén listos
    const loadAssetData = async (assetsReady) => {
        // Cargar feedback de texto si está listo y no lo hemos cargado ya
        if (assetsReady.feedback && !analysisState.assets.feedback.data) {
            try {
                console.log('📝 Loading feedback text...');
                const response = await fetch(`http://localhost:8000/assets/${jobId}/feedback.txt`);
                if (response.ok) {
                    const feedbackText = await response.text();
                    console.log('✅ Feedback loaded');
                    setAnalysisState(prev => ({
                        ...prev,
                        assets: {
                            ...prev.assets,
                            feedback: {
                                ...prev.assets.feedback,
                                data: feedbackText,
                                error: null
                            }
                        }
                    }));
                }
            } catch (error) {
                console.error('❌ Error loading feedback:', error);
                setAnalysisState(prev => ({
                    ...prev,
                    assets: {
                        ...prev.assets,
                        feedback: {
                            ...prev.assets.feedback,
                            error: 'Error cargando feedback'
                        }
                    }
                }));
            }
        }

        // TODO: Aquí cargaremos los datos del radar chart más tarde
        // Por ahora mantenemos los datos mock en el componente FeedbackRadarChart
    };

    // Iniciar polling
    const startPolling = () => {
        console.log(`🚀 Starting polling for job ${jobId}`);
        // Hacer primera consulta inmediatamente
        pollJobStatus();
        
        // Configurar polling cada 3 segundos
        pollingIntervalRef.current = setInterval(pollJobStatus, 3000);
    };

    // Parar polling
    const stopPolling = () => {
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
            console.log('⏹️ Polling stopped');
        }
    };

    // Efecto para iniciar/parar polling
    useEffect(() => {
        if (jobId) {
            startPolling();
        }

        // Cleanup al desmontar el componente
        return () => {
            stopPolling();
        };
    }, [jobId]);

    const handleBackToExercises = () => {
        // Limpiar polling antes de navegar
        stopPolling();
        
        // Opcional: limpiar archivos del servidor
        if (jobId) {
            console.log(`🧹 Cleaning up job ${jobId}`);
            fetch(`http://localhost:8000/jobs/${jobId}`, { method: 'DELETE' })
                .catch(error => console.warn('⚠️ Error cleaning up job:', error));
        }
        
        navigate(-1);
    };

    const handleDownloadReport = async () => {
        if (!analysisState.assets.report.ready) return;
        
        try {
            console.log('📄 Downloading report...');
            const response = await fetch(`http://localhost:8000/assets/${jobId}/report.json`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `${exerciseName.replace(/\s+/g, '_')}_report.json`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                window.URL.revokeObjectURL(url);
                console.log('✅ Report downloaded');
            }
        } catch (error) {
            console.error('❌ Error downloading report:', error);
            alert('Error descargando el reporte');
        }
    };

    // Mostrar error si no hay jobId
    if (!jobId) {
        return (
            <div className="feedback-analysis-container">
                <div style={{ textAlign: 'center', padding: '40px' }}>
                    <h2 style={{ color: 'var(--font-color)' }}>Error: No se encontró el análisis</h2>
                    <p style={{ color: 'var(--font-color)' }}>No hay un job_id válido para mostrar resultados.</p>
                    <button className="feedback-back-button" onClick={() => navigate(-1)}>
                        <MdArrowBack size={24} />
                        <span>Volver</span>
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="feedback-analysis-container">
            {/* Header con título y botón de volver */}
            <div className="feedback-header">
                <div className="feedback-title-section">
                    <h1 className="feedback-exercise-title">{exerciseName}</h1>
                    <div className="feedback-status-info">
                        <span className="feedback-current-step">
                            {analysisState.status === 'processing' ? `Procesando: ${analysisState.currentStep}` :
                             analysisState.status === 'completed' ? 'Análisis completado' :
                             analysisState.status === 'error' ? 'Error en el análisis' : 'Iniciando...'}
                        </span>
                        {analysisState.completedSteps.length > 0 && (
                            <span className="feedback-completed-steps">
                                Pasos completados: {analysisState.completedSteps.length}
                            </span>
                        )}
                    </div>
                </div>
                <div className="feedback-back-section">
                    <button 
                        className="feedback-back-button"
                        onClick={handleBackToExercises}
                        title="Volver a ejercicios"
                    >
                        <MdArrowBack size={24} />
                        <span>Volver</span>
                    </button>
                </div>
            </div>

            {/* Mostrar error general si existe */}
            {analysisState.error && (
                <div className="feedback-error-banner">
                    <p>❌ {analysisState.error}</p>
                </div>
            )}

            {/* Grid principal de dos columnas */}
            <div className="feedback-main-grid">
                {/* Columna izquierda */}
                <div className="feedback-left-column">
                    {/* Chat de feedback */}
                    <div className="feedback-chat-section">
                        <FeedbackChat 
                            isLoading={analysisState.assets.feedback.loading}
                            feedbackText={analysisState.assets.feedback.data}
                            error={analysisState.assets.feedback.error}
                            currentStep={analysisState.currentStep}
                        />
                    </div>

                    {/* Video del análisis */}
                    <div className="feedback-video-section">
                        <FeedbackVideoPlayer 
                            isLoading={analysisState.assets.video.loading}
                            videoReady={analysisState.assets.video.ready}
                            jobId={jobId}
                            error={analysisState.assets.video.error}
                        />
                    </div>
                </div>

                {/* Columna derecha */}
                <div className="feedback-right-column">
                    {/* Gráfico radar */}
                    <div className="feedback-radar-section">
                        <FeedbackRadarChart 
                            isLoading={analysisState.assets.radar.loading}
                            radarData={analysisState.assets.radar.data}
                            error={analysisState.assets.radar.error}
                        />
                    </div>

                    {/* Botón de descarga de reporte */}
                    <div className="feedback-report-section">
                        <button 
                            className="feedback-download-button"
                            disabled={!analysisState.assets.report.ready}
                            onClick={handleDownloadReport}
                            title="Descargar reporte completo"
                        >
                            {analysisState.assets.report.loading ? (
                                <>
                                    <div className="loading-spinner"></div>
                                    Generando reporte...
                                </>
                            ) : analysisState.assets.report.ready ? (
                                '📄 Descargar Reporte JSON'
                            ) : (
                                'Reporte no disponible'
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FeedbackAnalysisGrid;