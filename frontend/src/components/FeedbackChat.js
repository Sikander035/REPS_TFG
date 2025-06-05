import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const FeedbackChat = ({ isLoading, feedbackText, error, currentStep }) => {
    // Estados para la animación de escritura
    const [displayedText, setDisplayedText] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [hasStartedTyping, setHasStartedTyping] = useState(false);
    const typingIntervalRef = useRef(null);
    const typingSpeedRef = useRef(20); // milisegundos entre caracteres (ajustable)
    
    // Función mejorada para limpiar el texto eliminando notas entre paréntesis al final
    const cleanFeedbackText = (text) => {
        if (!text) return text;
        
        // Buscar patrones de notas al final entre paréntesis o asteriscos
        // Incluye múltiples variaciones de notas del LLM
        const notePatterns = [
            /\s*\*?\s*\([^)]*[Nn]ota[^)]*\)\s*\.?\s*\.?\s*$/,  // *(Nota: ...)
            /\s*\*?\s*\([^)]*no se mencionan[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... no se mencionan ...)
            /\s*\*?\s*\([^)]*se evitan[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... se evitan ...)
            /\s*\*?\s*\([^)]*disclaimer[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(Disclaimer: ...)
            /\s*\*?\s*\([^)]*aclaración[^)]*\)\s*\.?\s*\.?\s*$/i,   // *(Aclaración: ...)
            /\s*\*?\s*\([^)]*notas explicativas[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... notas explicativas ...)
            /\s*\*?\s*\([^)]*preguntas adicionales[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... preguntas adicionales ...)
            /\s*\*?\s*\([^)]*tal como se solicita[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... tal como se solicita ...)
            /\s*\*?\s*\([^)]*como se solicita[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(... como se solicita ...)
            /\s*\*?\s*\([^)]*no incluyo[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(No incluyo ...)
            /\s*\*?\s*\([^)]*evito[^)]*\)\s*\.?\s*\.?\s*$/i,  // *(Evito ...)
            /\s*\*?\s*\([^)]*según las instrucciones[^)]*\)\s*\.?\s*\.?\s*$/i   // *(... según las instrucciones ...)
        ];
        
        let cleanedText = text;
        
        // Intentar cada patrón
        for (const pattern of notePatterns) {
            if (pattern.test(cleanedText)) {
                const originalLength = cleanedText.length;
                const matchedText = cleanedText.match(pattern)[0];
                cleanedText = cleanedText.replace(pattern, '').trim();
                
                console.log('🧹 Nota eliminada del feedback:', {
                    matchedText: matchedText.substring(0, 80) + '...',
                    originalLength,
                    cleanedLength: cleanedText.length,
                    removed: originalLength - cleanedText.length
                });
                
                break; // Solo eliminar una nota por vez
            }
        }
        
        // Si no termina en punto después de la limpieza, añadir uno
        if (cleanedText && !cleanedText.endsWith('.') && !cleanedText.endsWith('!') && !cleanedText.endsWith('?')) {
            cleanedText += '.';
        }
        
        return cleanedText;
    };
    
    // Detectar cuando llega el feedback y empezar a escribir
    useEffect(() => {
        console.log('🔍 FeedbackChat useEffect triggered:', {
            feedbackText: !!feedbackText,
            hasStartedTyping,
            error: !!error,
            textLength: feedbackText?.length || 0
        });
        
        // Iniciar typewriter si hay texto y no ha empezado
        if (feedbackText && feedbackText.length > 0 && !hasStartedTyping && !error) {
            console.log('🎯 Starting typewriter effect...');
            
            // Limpiar el texto antes de procesarlo
            const cleanedText = cleanFeedbackText(feedbackText);
            console.log('🧹 Text cleaned:', {
                originalLength: feedbackText.length,
                cleanedLength: cleanedText.length,
                removed: feedbackText.length - cleanedText.length
            });
            
            setIsTyping(true);
            setHasStartedTyping(true);
            setDisplayedText(''); // Asegurar que empiece vacío
            
            // Usar setTimeout para asegurar que el estado se actualice primero
            setTimeout(() => {
                startTypewriterEffect(cleanedText);
            }, 50);
        }
        
        // Cleanup cuando el componente se desmonta o cambia el feedback
        return () => {
            if (typingIntervalRef.current) {
                clearTimeout(typingIntervalRef.current);
            }
        };
    }, [feedbackText, hasStartedTyping, error]);

    // Reiniciar estados cuando se inicia un nuevo análisis
    useEffect(() => {
        if (isLoading && !feedbackText) {
            console.log('🔄 Resetting typewriter states...');
            setDisplayedText('');
            setIsTyping(false);
            setHasStartedTyping(false);
            if (typingIntervalRef.current) {
                clearTimeout(typingIntervalRef.current);
            }
        }
    }, [isLoading, feedbackText]);

    const startTypewriterEffect = (fullText) => {
        console.log('⌨️ Starting typewriter with text:', fullText.substring(0, 50) + '...');
        let currentIndex = 0;
        
        const typeNextCharacter = () => {
            if (currentIndex < fullText.length) {
                const nextChar = fullText[currentIndex];
                console.log(`📝 Typing character ${currentIndex}: "${nextChar}"`);
                
                setDisplayedText(prev => {
                    const newText = prev + nextChar;
                    return newText;
                });
                
                currentIndex++;
                
                // Ajustar velocidad según el carácter
                let speed = typingSpeedRef.current;
                
                // Pausas más largas después de puntos y saltos de línea
                if (nextChar === '.' || nextChar === '!' || nextChar === '?') {
                    speed = speed * 3;
                } else if (nextChar === ',' || nextChar === ';') {
                    speed = speed * 1.5;
                } else if (nextChar === '\n') {
                    speed = speed * 2;
                } else if (nextChar === ' ') {
                    speed = speed * 0.8;
                }
                
                typingIntervalRef.current = setTimeout(typeNextCharacter, speed);
            } else {
                // Terminó de escribir
                console.log('✅ Typewriter effect completed');
                setIsTyping(false);
            }
        };
        
        // Empezar a escribir
        typeNextCharacter();
    };

    const skipTypewriterEffect = () => {
        if (isTyping && feedbackText) {
            console.log('⏭️ Skipping typewriter effect');
            if (typingIntervalRef.current) {
                clearTimeout(typingIntervalRef.current);
            }
            // Usar el texto limpio al saltar la animación
            const cleanedText = cleanFeedbackText(feedbackText);
            setDisplayedText(cleanedText);
            setIsTyping(false);
        }
    };
    
    // Determinar el mensaje a mostrar según el estado
    const getDisplayMessage = () => {
        if (error) {
            return `Lo siento, hubo un error generando el feedback: ${error}`;
        }
        
        if (feedbackText) {
            // Limpiar el texto para mostrar
            const cleanedText = cleanFeedbackText(feedbackText);
            
            // Si está escribiendo o ya empezó a escribir, mostrar el texto parcial
            if (hasStartedTyping) {
                console.log('📖 Showing partial text - typing in progress');
                return displayedText;
            }
            // Si no ha empezado a escribir, mostrar el texto limpio completo (fallback)
            return cleanedText;
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

    // Componentes personalizados para el renderizado de markdown
    const markdownComponents = {
        // Headers con estilos personalizados
        h1: ({children, ...props}) => <h1 className="feedback-md-h1" {...props}>{children}</h1>,
        h2: ({children, ...props}) => <h2 className="feedback-md-h2" {...props}>{children}</h2>,
        h3: ({children, ...props}) => <h3 className="feedback-md-h3" {...props}>{children}</h3>,
        
        // Párrafos con espaciado mejorado
        p: ({children, ...props}) => <p className="feedback-md-paragraph" {...props}>{children}</p>,
        
        // Listas con mejor estilo
        ul: ({children, ...props}) => <ul className="feedback-md-list" {...props}>{children}</ul>,
        ol: ({children, ...props}) => <ol className="feedback-md-ordered-list" {...props}>{children}</ol>,
        li: ({children, ...props}) => <li className="feedback-md-list-item" {...props}>{children}</li>,
        
        // Texto en negrita y cursiva
        strong: ({children, ...props}) => <strong className="feedback-md-strong" {...props}>{children}</strong>,
        em: ({children, ...props}) => <em className="feedback-md-emphasis" {...props}>{children}</em>,
        
        // Código inline y bloques de código
        code: ({inline, children, ...props}) => 
            inline ? 
                <code className="feedback-md-code-inline" {...props}>{children}</code> : 
                <pre className="feedback-md-code-block" {...props}><code>{children}</code></pre>,
        
        // Blockquotes para citas
        blockquote: ({children, ...props}) => <blockquote className="feedback-md-blockquote" {...props}>{children}</blockquote>,
        
        // Enlaces (si los hay)
        a: ({href, children, ...props}) => <a href={href} className="feedback-md-link" target="_blank" rel="noopener noreferrer" {...props}>{children}</a>,
        
        // Separadores
        hr: (props) => <hr className="feedback-md-divider" {...props} />,
        
        // Tablas (por si el feedback incluye tablas)
        table: ({children, ...props}) => <table className="feedback-md-table" {...props}>{children}</table>,
        thead: ({children, ...props}) => <thead className="feedback-md-table-head" {...props}>{children}</thead>,
        tbody: ({children, ...props}) => <tbody className="feedback-md-table-body" {...props}>{children}</tbody>,
        tr: ({children, ...props}) => <tr className="feedback-md-table-row" {...props}>{children}</tr>,
        th: ({children, ...props}) => <th className="feedback-md-table-header" {...props}>{children}</th>,
        td: ({children, ...props}) => <td className="feedback-md-table-cell" {...props}>{children}</td>,
    };

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
                <div className="feedback-chat-bubble" onClick={skipTypewriterEffect}>
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
                            {error ? (
                                <p className="feedback-error-message">{displayMessage}</p>
                            ) : feedbackText ? (
                                <div className="feedback-markdown-container">
                                    {/* Mostrar contenido de markdown */}
                                    {displayedText.length > 0 ? (
                                        <ReactMarkdown 
                                            components={markdownComponents}
                                            remarkPlugins={[remarkGfm]}
                                        >
                                            {displayedText}
                                        </ReactMarkdown>
                                    ) : isTyping ? (
                                        // Mostrar solo cursor si está empezando a escribir
                                        <div className="feedback-starting-to-type">
                                            <span className="feedback-typing-cursor">|</span>
                                        </div>
                                    ) : (
                                        // Fallback: mostrar todo el texto limpio
                                        <ReactMarkdown 
                                            components={markdownComponents}
                                            remarkPlugins={[remarkGfm]}
                                        >
                                            {cleanFeedbackText(feedbackText)}
                                        </ReactMarkdown>
                                    )}
                                    
                                    {/* Cursor de escritura cuando está escribiendo y hay texto */}
                                    {isTyping && displayedText.length > 0 && (
                                        <span className="feedback-typing-cursor">|</span>
                                    )}
                                    
                                    {/* Hint para saltar la animación */}
                                    {isTyping && (
                                        <div className="feedback-skip-hint">
                                            <small>Haz clic para ver el mensaje completo</small>
                                        </div>
                                    )}
                                </div>
                            ) : (
                                <p className="feedback-placeholder-message">{displayMessage}</p>
                            )}
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
                     feedbackText ? (isTyping ? 'Escribiendo...' : 'Análisis completado') :
                     isLoading ? 'Analizando...' : 'En espera'}
                </span>
            </div>
        </div>
    );
};

export default FeedbackChat;