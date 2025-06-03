import React from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import FeedbackAnalysisGrid from '../components/FeedbackAnalysisGrid';

const Feedback = () => {
    const location = useLocation();
    // TODO: Aquí recibiremos el job_id en lugar del archivo original
    const { originalFile, fileName, jobId, exerciseName } = location.state || {};

    // TODO: Placeholder data - reemplazar con datos reales del análisis
    const exerciseDisplayName = exerciseName || "Military Press";

    if (!originalFile && !jobId) {
        return (
            <div className='page-container'>
                <Navbar />
                <div className='main-content' style={{ textAlign: 'center', padding: '40px' }}>
                    <h2 style={{ color: 'var(--font-color)' }}>No hay análisis para mostrar</h2>
                    <p style={{ color: 'var(--font-color)' }}>Por favor, regresa y sube un video primero.</p>
                </div>
                <Footer />
            </div>
        );
    }

    return (
        <div className='page-container'>
            <Navbar />
            <div className='main-content'>
                <FeedbackAnalysisGrid 
                    exerciseName={exerciseDisplayName}
                    originalFile={originalFile}
                    fileName={fileName}
                    jobId={jobId}
                />
            </div>
            <Footer />
        </div>
    );
};

export default Feedback;