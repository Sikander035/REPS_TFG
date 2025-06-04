import React from 'react';

const FeedbackChat = ({ isLoading, feedbackText, error, currentStep }) => {
    
    // Determinar el mensaje a mostrar según el estado
    const getDisplayMessage = () => {
        if (error) {
            return `Lo siento, hubo un error generando el feedback: ${error}`;
        }
        
        if (feedbackText) {
            return feedbackText;
        }
        
        if (isLoading) {
            return null; // Se mostrará la animación de puntos
        }
        
        return "Esperando que el análisis comience...";
    };

    // Obtener mensaje de estado para la animación de carga
    const getLoadingMessage = () => {
        switch (currentStep) {
            case 'extraction':
                return 'Extrayendo landmarks del video...';
            case 'load_expert':
                return 'Cargando datos del experto...';
            case 'repetition_detection':
                return 'Detectando repeticiones...';
            case 'synchronization':
                return 'Sincronizando datos...';
            case 'normalization':
                return 'Normalizando esqueletos...';
            case 'alignment':
                return 'Alineando datos...';
            case 'analysis':
                return 'Analizando tu técnica...';
            case 'feedback_generation':
                return 'Generando feedback personalizado...';
            case 'generating_assets':
                return 'Generando recursos finales...';
            default:
                return 'Analizando tu técnica...';
        }
    };

    const displayMessage = getDisplayMessage();

    return (
        <div className="feedback-chat-container">
            <div className="feedback-chat-header">
                <h3>Análisis de IA</h3>
            </div>
            
            <div className="feedback-chat-content">
                {/* Avatar del entrenador IA */}
                <div className="feedback-chat-avatar">
                    <img 
                        src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face" 
                        alt="Entrenador IA" 
                        className="feedback-avatar-image"
                    />
                </div>

                {/* Bocadillo de mensaje */}
                <div className="feedback-chat-bubble">
                    {isLoading && !feedbackText ? (
                        <div className="feedback-typing-indicator">
                            <div className="feedback-typing-dots">
                                <span className="dot"></span>
                                <span className="dot"></span>
                                <span className="dot"></span>
                            </div>
                            <span className="feedback-typing-text">{getLoadingMessage()}</span>
                        </div>
                    ) : (
                        <div className="feedback-message-content">
                            <p className={error ? 'feedback-error-message' : ''}>{displayMessage}</p>
                        </div>
                    )}
                    
                    {/* Flecha del bocadillo */}
                    <div className="feedback-bubble-arrow"></div>
                </div>
            </div>

            {/* Información del entrenador */}
            <div className="feedback-trainer-info">
                <span className="feedback-trainer-name">Coach IA</span>
                <span className="feedback-trainer-status">
                    {error ? 'Error en el análisis' :
                     feedbackText ? 'Análisis completado' :
                     isLoading ? 'Analizando...' : 'En espera'}
                </span>
            </div>
        </div>
    );
};

export default FeedbackChat;