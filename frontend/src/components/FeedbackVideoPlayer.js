import React from 'react';
import ReactPlayer from 'react-player';

const FeedbackVideoPlayer = ({ isLoading, videoReady, jobId, error }) => {
    
    // Construir URL real del video comparativo cuando esté listo
    const getVideoUrl = () => {
        if (jobId && videoReady) {
            return `http://localhost:8000/assets/${jobId}/video.mp4`;
        }
        return null;
    };

    // Determinar el estado a mostrar
    const getVideoState = () => {
        if (error) {
            return 'error';
        }
        if (videoReady) {
            return 'ready';
        }
        if (isLoading) {
            return 'loading';
        }
        return 'waiting';
    };

    const videoState = getVideoState();
    const videoUrl = getVideoUrl();

    return (
        <div className="feedback-video-container">
            <div className="feedback-video-header">
                <h3>Video Comparativo</h3>
                <span className="feedback-video-status">
                    {videoState === 'loading' && 'Generando...'}
                    {videoState === 'ready' && 'Listo'}
                    {videoState === 'error' && 'Error'}
                    {videoState === 'waiting' && 'En espera'}
                </span>
            </div>

            <div className="feedback-video-content">
                {videoState === 'loading' && (
                    // Estado de carga - cuadro oscuro con spinner
                    <div className="feedback-video-loading">
                        <div className="loading-spinner"></div>
                        <p>Generando video comparativo...</p>
                        <small>Este proceso puede tomar varios minutos</small>
                    </div>
                )}

                {videoState === 'ready' && videoUrl && (
                    // Video listo - mostrar player
                    <div className="feedback-video-player-wrapper">
                        <ReactPlayer
                            className="feedback-video-player"
                            url={videoUrl}
                            controls={true}
                            width="100%"
                            height="100%"
                            light={false}
                            playing={false}
                            config={{
                                file: {
                                    attributes: {
                                        controlsList: 'nodownload'
                                    }
                                }
                            }}
                            onError={(error) => {
                                console.error('Error loading video:', error);
                            }}
                            onReady={() => {
                                console.log('✅ Video loaded successfully');
                            }}
                        />
                    </div>
                )}

                {videoState === 'error' && (
                    // Estado de error
                    <div className="feedback-video-placeholder">
                        <p>❌ Error generando el video</p>
                        <small>{error}</small>
                    </div>
                )}

                {videoState === 'waiting' && (
                    // Estado de espera
                    <div className="feedback-video-placeholder">
                        <p>⏳ Esperando a que comience la generación del video</p>
                    </div>
                )}
            </div>

            {/* Información adicional */}
            {videoReady && (
                <div className="feedback-video-info">
                    <span className="feedback-video-description">
                        Comparación entre tu técnica (izquierda) y la técnica ideal (derecha)
                    </span>
                </div>
            )}
        </div>
    );
};

export default FeedbackVideoPlayer;