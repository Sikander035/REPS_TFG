import React from 'react';

const FeedbackChat = ({ isLoading, feedbackText }) => {
    return (
        <div className="feedback-chat-container">
            <div className="feedback-chat-header">
                <h3>Análisis de IA</h3>
            </div>
            
            <div className="feedback-chat-content">
                {/* Avatar del entrenador IA */}
                <div className="feedback-chat-avatar">
                    {/* TODO: Reemplazar con imagen real del avatar del entrenador IA */}
                    <img 
                        src="https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=150&h=150&fit=crop&crop=face" 
                        alt="Entrenador IA" 
                        className="feedback-avatar-image"
                    />
                </div>

                {/* Bocadillo de mensaje */}
                <div className="feedback-chat-bubble">
                    {isLoading ? (
                        <div className="feedback-typing-indicator">
                            <div className="feedback-typing-dots">
                                <span className="dot"></span>
                                <span className="dot"></span>
                                <span className="dot"></span>
                            </div>
                            <span className="feedback-typing-text">Analizando tu técnica...</span>
                        </div>
                    ) : (
                        <div className="feedback-message-content">
                            <p>{feedbackText || "Esperando análisis..."}</p>
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
                    {isLoading ? 'Analizando...' : 'Análisis completado'}
                </span>
            </div>
        </div>
    );
};

export default FeedbackChat;