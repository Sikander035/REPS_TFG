import React, { useState, useRef, useEffect } from 'react';
import ReactPlayer from 'react-player';

const FeedbackVideoPlayer = ({ isLoading, videoReady, jobId, error }) => {
    const [videoError, setVideoError] = useState(null);
    const [videoInfo, setVideoInfo] = useState({});
    const [debugInfo, setDebugInfo] = useState({});
    const videoRef = useRef(null);
    const reactPlayerRef = useRef(null);

    // URL del video
    const videoUrl = videoReady ? `http://localhost:8000/assets/${jobId}/video.mp4` : null;

    // Debug: verificar video cuando esté listo
    useEffect(() => {
        if (videoReady && videoUrl) {
            console.log('🎥 Video URL:', videoUrl);
            
            // Hacer petición HEAD para verificar headers
            fetch(videoUrl, { method: 'HEAD' })
                .then(response => {
                    console.log('📊 Video HEAD response:', {
                        status: response.status,
                        headers: Object.fromEntries(response.headers.entries()),
                        url: response.url
                    });
                    
                    setDebugInfo({
                        status: response.status,
                        contentType: response.headers.get('content-type'),
                        contentLength: response.headers.get('content-length'),
                        acceptRanges: response.headers.get('accept-ranges'),
                        lastModified: response.headers.get('last-modified'),
                    });
                })
                .catch(err => {
                    console.error('❌ Error verificando video:', err);
                    setVideoError(`Error verificando video: ${err.message}`);
                });
        }
    }, [videoReady, videoUrl]);

    // Handlers de eventos para debugging
    const handleVideoLoad = () => {
        console.log('✅ Video cargado exitosamente');
        setVideoError(null);
        
        if (videoRef.current) {
            const video = videoRef.current;
            setVideoInfo({
                duration: video.duration,
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                readyState: video.readyState,
                networkState: video.networkState,
                currentSrc: video.currentSrc,
            });
            console.log('📊 Video info:', video);
        }
    };

    const handleVideoError = (e) => {
        console.error('❌ Video error:', e);
        
        let errorMessage = 'Error desconocido del video';
        
        if (videoRef.current && videoRef.current.error) {
            const error = videoRef.current.error;
            const errorCodes = {
                1: 'MEDIA_ERR_ABORTED - Proceso abortado por el usuario',
                2: 'MEDIA_ERR_NETWORK - Error de red',
                3: 'MEDIA_ERR_DECODE - Error de decodificación',
                4: 'MEDIA_ERR_SRC_NOT_SUPPORTED - Formato no soportado'
            };
            
            errorMessage = errorCodes[error.code] || `Error código ${error.code}`;
            console.error('📺 Video element error:', {
                code: error.code,
                message: error.message,
                details: errorMessage
            });
        }
        
        setVideoError(errorMessage);
    };

    const handleReactPlayerError = (error) => {
        console.error('❌ ReactPlayer error:', error);
        setVideoError(`ReactPlayer: ${error.message || 'Error desconocido'}`);
    };

    const handleVideoProgress = (progress) => {
        console.log('⏯️ Video progress:', progress);
    };

    const handleVideoReady = () => {
        console.log('✅ ReactPlayer ready');
        if (reactPlayerRef.current) {
            const player = reactPlayerRef.current.getInternalPlayer();
            console.log('📊 ReactPlayer internal:', player);
        }
    };

    // Función para probar descarga directa
    const testDirectDownload = () => {
        if (videoUrl) {
            console.log('🔄 Probando descarga directa...');
            window.open(videoUrl, '_blank');
        }
    };

    // Función para probar en nueva pestaña
    const testInNewTab = () => {
        if (videoUrl) {
            console.log('🔄 Abriendo video en nueva pestaña...');
            window.open(videoUrl, '_blank');
        }
    };

    const resetPlayer = () => {
        console.log('🔄 Reiniciando player...');
        setVideoError(null);
        setVideoInfo({});
        if (videoRef.current) {
            videoRef.current.load();
        }
    };

    return (
        <div className="feedback-video-container">
            <div className="feedback-video-header">
                <h3>Video de Análisis</h3>
                <span className="feedback-video-status">
                    {isLoading ? 'Generando...' : 
                     videoReady ? (videoError ? 'Error' : 'Listo') : 'Pendiente'}
                </span>
            </div>

            <div className="feedback-video-content">
                {isLoading ? (
                    <div className="feedback-video-loading">
                        <div className="loading-spinner"></div>
                        <p>Generando video comparativo...</p>
                        <small>Esto puede tomar varios minutos</small>
                    </div>
                ) : videoReady ? (
                    <>
                        {/* Debug Info Panel */}
                        <div style={{ 
                            backgroundColor: 'rgba(0,0,0,0.8)', 
                            padding: '10px', 
                            borderRadius: '4px', 
                            marginBottom: '10px',
                            fontSize: '12px',
                            color: '#ccc'
                        }}>
                            <strong>🔍 Debug Info:</strong>
                            <br />URL: {videoUrl}
                            <br />Status: {debugInfo.status} | Type: {debugInfo.contentType}
                            <br />Size: {debugInfo.contentLength ? `${(debugInfo.contentLength / 1024 / 1024).toFixed(1)}MB` : 'Unknown'}
                            <br />Ranges: {debugInfo.acceptRanges} | Duration: {videoInfo.duration ? `${videoInfo.duration.toFixed(1)}s` : 'Unknown'}
                            
                            <div style={{ marginTop: '8px', display: 'flex', gap: '8px' }}>
                                <button onClick={testDirectDownload} style={{
                                    padding: '4px 8px', fontSize: '10px', backgroundColor: '#666', 
                                    color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer'
                                }}>
                                    📥 Descargar
                                </button>
                                <button onClick={testInNewTab} style={{
                                    padding: '4px 8px', fontSize: '10px', backgroundColor: '#666', 
                                    color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer'
                                }}>
                                    🔗 Nueva pestaña
                                </button>
                                <button onClick={resetPlayer} style={{
                                    padding: '4px 8px', fontSize: '10px', backgroundColor: '#666', 
                                    color: 'white', border: 'none', borderRadius: '3px', cursor: 'pointer'
                                }}>
                                    🔄 Reset
                                </button>
                            </div>
                        </div>

                        {videoError ? (
                            <div style={{
                                padding: '20px',
                                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                                border: '1px solid #ff6b6b',
                                borderRadius: '8px',
                                color: '#ff6b6b',
                                textAlign: 'center'
                            }}>
                                <p><strong>❌ Error de video:</strong></p>
                                <p>{videoError}</p>
                                <small>Verifica la consola del navegador para más detalles</small>
                            </div>
                        ) : (
                            <div className="feedback-video-player-wrapper">
                                {/* Probar HTML5 video nativo primero */}
                                <div style={{ marginBottom: '10px' }}>
                                    <strong>🎬 HTML5 Video Native:</strong>
                                    <video
                                        ref={videoRef}
                                        width="100%"
                                        height="200"
                                        controls
                                        preload="metadata"
                                        crossOrigin="anonymous"
                                        onLoadedData={handleVideoLoad}
                                        onError={handleVideoError}
                                        onLoadStart={() => console.log('🔄 Video load start')}
                                        onLoadedMetadata={() => console.log('📊 Video metadata loaded')}
                                        onCanPlay={() => console.log('▶️ Video can play')}
                                        onCanPlayThrough={() => console.log('⏯️ Video can play through')}
                                        style={{
                                            border: '2px solid var(--secondary-color)',
                                            borderRadius: '8px',
                                            backgroundColor: '#000'
                                        }}
                                    >
                                        <source src={videoUrl} type="video/mp4" />
                                        Tu navegador no soporta el elemento video.
                                    </video>
                                </div>

                                {/* ReactPlayer como alternativa */}
                                <div style={{ marginTop: '10px' }}>
                                    <strong>🎭 ReactPlayer:</strong>
                                    <div style={{ marginTop: '5px' }}>
                                        <ReactPlayer
                                            ref={reactPlayerRef}
                                            url={videoUrl}
                                            controls={true}
                                            width="100%"
                                            height="200px"
                                            onReady={handleVideoReady}
                                            onError={handleReactPlayerError}
                                            onProgress={handleVideoProgress}
                                            onStart={() => console.log('▶️ ReactPlayer started')}
                                            onBuffer={() => console.log('⏳ ReactPlayer buffering')}
                                            onBufferEnd={() => console.log('✅ ReactPlayer buffer end')}
                                            config={{
                                                file: {
                                                    attributes: {
                                                        crossOrigin: 'anonymous',
                                                        preload: 'metadata'
                                                    }
                                                }
                                            }}
                                            style={{
                                                border: '2px solid var(--secondary-color)',
                                                borderRadius: '8px'
                                            }}
                                        />
                                    </div>
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="feedback-video-placeholder">
                        <p>Video no disponible</p>
                        <small>El video se generará durante el análisis</small>
                    </div>
                )}
            </div>

            {videoReady && !videoError && (
                <div className="feedback-video-info">
                    <p className="feedback-video-description">
                        Video comparativo con tu técnica (rojo) vs técnica ideal (azul)
                    </p>
                </div>
            )}
        </div>
    );
};

export default FeedbackVideoPlayer;