import React from 'react';
import { useLocation } from 'react-router-dom';
import ReactPlayer from 'react-player';

const Feedback = () => {
    const location = useLocation();
    const { originalFile, fileName } = location.state || {};

    if (!originalFile) {
        return (
            <div style={{ padding: '20px', textAlign: 'center' }}>
                <h2>No hay video para mostrar</h2>
                <p>Por favor, regresa y sube un video primero.</p>
            </div>
        );
    }

    return (
        <div className="feedback-page">
            <h1 className="section-title">Análisis de tu ejercicio</h1>
            <div className="feedback-content">
                <div className="video-section">
                    <h3>Video original: {fileName}</h3>
                    <ReactPlayer
                        url={URL.createObjectURL(originalFile)}
                        controls
                        width="100%"
                        height="400px"
                    />
                </div>
                <div className="feedback-details">
                    <h3>Resultados del análisis</h3>
                    <p>Aquí aparecerán los detalles del feedback una vez procesado el video.</p>
                </div>
            </div>
        </div>
    );
};

export default Feedback;