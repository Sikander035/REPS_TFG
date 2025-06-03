import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MdArrowBack } from 'react-icons/md';
import FeedbackChat from './FeedbackChat';
import FeedbackVideoPlayer from './FeedbackVideoPlayer';
import FeedbackRadarChart from './FeedbackRadarChart';

const FeedbackAnalysisGrid = ({ exerciseName, originalFile, fileName, jobId }) => {
    const navigate = useNavigate();
    
    // Estados para simular la carga progresiva de assets
    const [analysisState, setAnalysisState] = useState({
        feedback: { ready: false, loading: true, data: null },
        video: { ready: false, loading: true, data: null },
        radar: { ready: false, loading: true, data: null },
        report: { ready: false, loading: true, data: null }
    });

    // TODO: Reemplazar con polling real a la API
    useEffect(() => {
        // Simular carga progresiva de assets
        const simulateAssetLoading = () => {
            // Radar se carga primero (después de 2 segundos)
            setTimeout(() => {
                setAnalysisState(prev => ({
                    ...prev,
                    radar: { ready: true, loading: false, data: getMockRadarData() }
                }));
            }, 2000);

            // Feedback de IA se carga después (4 segundos)
            setTimeout(() => {
                setAnalysisState(prev => ({
                    ...prev,
                    feedback: { ready: true, loading: false, data: getMockFeedbackData() }
                }));
            }, 4000);

            // Video comparativo se carga al final (6 segundos)
            setTimeout(() => {
                setAnalysisState(prev => ({
                    ...prev,
                    video: { ready: true, loading: false, data: "comparison_video.mp4" }
                }));
            }, 6000);
        };

        simulateAssetLoading();
    }, [jobId]);

    const handleBackToExercises = () => {
        // TODO: Añadir lógica de limpieza si es necesario
        navigate(-1); // Volver a la página anterior
    };

    // TODO: Datos mock - reemplazar con datos reales de la API
    const getMockRadarData = () => ([
        { metric: 'Postura', score: 85 },
        { metric: 'Tiempo', score: 78 },
        { metric: 'Amplitud', score: 92 },
        { metric: 'Estabilidad', score: 76 },
        { metric: 'Coordinación', score: 88 }
    ]);

    const getMockFeedbackData = () => 
        "¡Excelente trabajo en tu Military Press! Tu postura general es muy buena, mantienes la espalda recta durante todo el movimiento. Sin embargo, he notado que podrías mejorar la estabilidad en la fase de descenso. Te recomiendo trabajar más el core y reducir ligeramente el peso para perfeccionar la técnica antes de aumentar la carga.";

    return (
        <div className="feedback-analysis-container">
            {/* Header con título y botón de volver */}
            <div className="feedback-header">
                <div className="feedback-title-section">
                    <h1 className="feedback-exercise-title">{exerciseName}</h1>
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

            {/* Grid principal de dos columnas */}
            <div className="feedback-main-grid">
                {/* Columna izquierda */}
                <div className="feedback-left-column">
                    {/* Chat de feedback */}
                    <div className="feedback-chat-section">
                        <FeedbackChat 
                            isLoading={analysisState.feedback.loading}
                            feedbackText={analysisState.feedback.data}
                        />
                    </div>

                    {/* Video del análisis */}
                    <div className="feedback-video-section">
                        <FeedbackVideoPlayer 
                            isLoading={analysisState.video.loading}
                            videoReady={analysisState.video.ready}
                            videoData={analysisState.video.data}
                            originalFile={originalFile}
                            jobId={jobId}
                        />
                    </div>
                </div>

                {/* Columna derecha */}
                <div className="feedback-right-column">
                    {/* Gráfico radar */}
                    <div className="feedback-radar-section">
                        <FeedbackRadarChart 
                            isLoading={analysisState.radar.loading}
                            radarData={analysisState.radar.data}
                        />
                    </div>

                    {/* Botón de descarga de reporte (opcional) */}
                    <div className="feedback-report-section">
                        <button 
                            className="feedback-download-button"
                            disabled={analysisState.report.loading}
                            title="Descargar reporte completo"
                        >
                            {analysisState.report.loading ? (
                                <div className="loading-spinner"></div>
                            ) : (
                                'Descargar Reporte JSON'
                            )}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default FeedbackAnalysisGrid;