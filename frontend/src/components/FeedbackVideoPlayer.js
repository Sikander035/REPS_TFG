import React, { useState } from 'react';
import ReactPlayer from 'react-player';
import { MdFileDownload } from 'react-icons/md';

const FeedbackVideoPlayer = ({ isLoading, videoReady, jobId, error }) => {
    const [videoError, setVideoError] = useState(null);

    // URL del video
    const videoUrl = videoReady ? `http://localhost:8000/assets/${jobId}/video.mp4` : null;

    // Manejar errores del video
    const handleVideoError = (error) => {
        console.error('❌ Video error:', error);
        setVideoError('Error cargando el video de análisis');
    };

    // Función para descargar video
    const handleDownloadVideo = () => {
        if (videoUrl) {
            const link = document.createElement('a');
            link.href = videoUrl;
            link.download = 'analisis_comparativo.mp4';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };

    return (
        <div className="feedback-video-container">
            {/* Header simple - separado */}
            <div className="feedback-video-header">
                <h3>Video de Análisis</h3>
            </div>

            {/* Contenedor principal con fondo */}
            <div className="feedback-video-main">
                {isLoading ? (
                    // Estado de carga igual al del radar
                    <div className="feedback-video-loading">
                        <div className="loading-spinner"></div>
                        <p>Generando video comparativo...</p>
                    </div>
                ) : videoReady ? (
                    <>
                        {/* Contenido del video */}
                        <div className="feedback-video-content">
                            {videoError ? (
                                <div className="feedback-video-error">
                                    <p>❌ {videoError}</p>
                                </div>
                            ) : (
                                <div className="feedback-video-player-wrapper">
                                    <ReactPlayer
                                        url={videoUrl}
                                        controls={true}
                                        width="100%"
                                        height="100%"
                                        onError={handleVideoError}
                                        config={{
                                            file: {
                                                attributes: {
                                                    crossOrigin: 'anonymous',
                                                    preload: 'metadata'
                                                }
                                            }
                                        }}
                                    />
                                </div>
                            )}
                        </div>

                        {/* Footer con leyenda y botón de descarga */}
                        {!videoError && (
                            <div className="feedback-video-footer">
                                <div className="feedback-video-legend">
                                    <p>Video comparativo con tu técnica (verde) vs técnica ideal (rojo)</p>
                                </div>
                                <button 
                                    className="feedback-video-download-btn"
                                    onClick={handleDownloadVideo}
                                    title="Descargar video"
                                >
                                    <MdFileDownload size={20} />
                                    <span>Descargar</span>
                                </button>
                            </div>
                        )}
                    </>
                ) : (
                    // Video no disponible
                    <div className="feedback-video-placeholder">
                        <p>Video no disponible</p>
                    </div>
                )}
            </div>
        </div>
    );
};

export default FeedbackVideoPlayer;