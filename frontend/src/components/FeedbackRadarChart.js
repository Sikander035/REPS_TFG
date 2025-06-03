import React from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

const FeedbackRadarChart = ({ isLoading, radarData }) => {
    return (
        <div className="feedback-radar-container">
            <div className="feedback-radar-header">
                <h3>Análisis Técnico</h3>
                <span className="feedback-radar-status">
                    {isLoading ? 'Calculando...' : radarData ? 'Completado' : 'Pendiente'}
                </span>
            </div>

            <div className="feedback-radar-content">
                {isLoading ? (
                    // Estado de carga - cuadro oscuro con spinner
                    <div className="feedback-radar-loading">
                        <div className="loading-spinner"></div>
                        <p>Analizando métricas...</p>
                    </div>
                ) : radarData ? (
                    // Gráfico radar listo
                    <div className="feedback-radar-chart-wrapper">
                        <ResponsiveContainer width="100%" height="100%">
                            <RadarChart data={radarData} margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <PolarGrid stroke="var(--secondary-color)" strokeOpacity={0.3} />
                                <PolarAngleAxis 
                                    dataKey="metric" 
                                    tick={{ fill: 'var(--font-color)', fontSize: 12 }}
                                />
                                <PolarRadiusAxis 
                                    angle={90} 
                                    domain={[0, 100]} 
                                    tick={{ fill: 'var(--font-color)', fontSize: 10 }}
                                    strokeOpacity={0.3}
                                />
                                <Radar
                                    name="Puntuación"
                                    dataKey="score"
                                    stroke="var(--secondary-color)"
                                    fill="var(--secondary-color)"
                                    fillOpacity={0.2}
                                    strokeWidth={2}
                                />
                            </RadarChart>
                        </ResponsiveContainer>
                    </div>
                ) : (
                    // Estado de error o fallback
                    <div className="feedback-radar-placeholder">
                        <p>Datos no disponibles</p>
                    </div>
                )}
            </div>

            {/* Información adicional del radar */}
            {radarData && (
                <div className="feedback-radar-info">
                    <div className="feedback-radar-scores">
                        {radarData.map((item, index) => (
                            <div key={index} className="feedback-radar-score-item">
                                <span className="feedback-score-label">{item.metric}:</span>
                                <span className="feedback-score-value">{item.score}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default FeedbackRadarChart;