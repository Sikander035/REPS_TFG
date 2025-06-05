import React, { useState, useEffect } from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer } from 'recharts';

const FeedbackRadarChart = ({ isLoading, jobId, error }) => {
    const [radarData, setRadarData] = useState(null);
    const [loadingData, setLoadingData] = useState(false);
    const [dataError, setDataError] = useState(null);

    // Cargar datos del radar cuando esté disponible
    useEffect(() => {
        const loadRadarData = async () => {
            if (!jobId || isLoading) return;
            
            setLoadingData(true);
            setDataError(null);
            
            try {
                console.log(`📊 Loading radar data for job ${jobId}...`);
                const response = await fetch(`http://localhost:8000/assets/${jobId}/radar.json`);
                
                if (response.ok) {
                    const data = await response.json();
                    console.log('✅ Radar data loaded:', data);
                    setRadarData(data);
                    setDataError(null);
                } else {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
            } catch (error) {
                console.error('❌ Error loading radar data:', error);
                setDataError(`Error cargando datos: ${error.message}`);
            } finally {
                setLoadingData(false);
            }
        };

        // Solo cargar si no estamos en loading general y no tenemos datos ya
        if (!isLoading && !radarData && !loadingData) {
            loadRadarData();
        }
    }, [isLoading, jobId, radarData, loadingData]);

    // Determinar el estado a mostrar
    const showLoading = isLoading || loadingData;
    const showError = error || dataError;
    const showChart = !showLoading && !showError && radarData && radarData.length > 0;

    return (
        <div className="feedback-radar-container">
            <div className="feedback-radar-header">
                <h3>Análisis Técnico</h3>
                <span className="feedback-radar-status">
                    {showLoading ? 'Calculando...' : 
                     showChart ? 'Completado' : 
                     showError ? 'Error' : 'Pendiente'}
                </span>
            </div>

            <div className="feedback-radar-content">
                {showLoading ? (
                    // Estado de carga
                    <div className="feedback-radar-loading">
                        <div className="loading-spinner"></div>
                        <p>
                            {isLoading ? 'Analizando métricas...' : 'Cargando gráfico...'}
                        </p>
                    </div>
                ) : showError ? (
                    // Estado de error
                    <div className="feedback-radar-placeholder">
                        <p>❌ {showError}</p>
                    </div>
                ) : showChart ? (
                    // Gráfico radar con datos reales
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
                    // Estado placeholder
                    <div className="feedback-radar-placeholder">
                        <p>Datos no disponibles</p>
                    </div>
                )}
            </div>

            {/* Información adicional del radar - solo si hay datos */}
            {showChart && (
                <div className="feedback-radar-info">
                    <div className="feedback-radar-scores">
                        {radarData.map((item, index) => (
                            <div key={index} className="feedback-radar-score-item">
                                <span className="feedback-score-label">{item.metric}:</span>
                                <span className="feedback-score-value">{Math.round(item.score)}%</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default FeedbackRadarChart;