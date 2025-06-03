import React from 'react';
import ReactPlayer from 'react-player';

const FeedbackVideoPlayer = ({ isLoading, videoReady, videoData, originalFile, jobId }) => {
    // TODO: Construir URL real del video comparativo cuando esté listo
    const getVideoUrl = () => {
        if (jobId && videoReady) {
            return `http://localhost:8000/assets/${jobId}/video.mp4`;
        }
        return null;
    };

    return (
        <div className="feedback-video-container">
            <div className="feedback-video-header">
                <h3>Video Comparativo</h3>
                <span className="feedback-video-status">
                    {isLoading ? 'Generando...' : videoReady ? 'Listo' : 'Pendiente'}
                </span>
            </div>

            <div className="feedback-video-content">
                {isLoading ? (
                    // Estado de carga - cuadro oscuro con spinner
                    <div className="feedback-video-loading">
                        <div className="loading-spinner"></div>
                        <p>Generando video comparativo...</p>
                    </div>
                ) : videoReady ? (
                    // Video listo - mostrar player
                    <div className="feedback-video-player-wrapper">
                        <ReactPlayer
                            className="feedback-video-player"
                            url={getVideoUrl()}
                            controls={true}
                            width="100%"
                            height="100%"
                            light={false}
                            playing={false}
                        />
                    </div>
                ) : (
                    // Estado de error o fallback
                    <div className="feedback-video-placeholder">
                        <p>Video no disponible</p>
                    </div>
                )}
            </div>

            {/* Información adicional */}
            {videoReady && (
                <div className="feedback-video-info">
                    <span className="feedback-video-description">
                        Comparación entre tu técnica y la técnica ideal
                    </span>
                </div>
            )}
        </div>
    );
};

export default FeedbackVideoPlayer;