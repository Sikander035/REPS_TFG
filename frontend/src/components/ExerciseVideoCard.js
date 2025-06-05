// src/components/ExerciseVideoCard.js
import React from 'react';
import ReactPlayer from 'react-player';


const ExerciseVideoCard = ({ name, description, video_path }) => {
    return (
        <div className='exercise-video-card'>
            <div className='exercise-video-card-grid'>
                <h1 className='section-title'>{name}</h1>
                <p className='exercise-video-card-description'>{description}</p>
                <div className='exercise-video-card-content'>
                    <div className='exercise-video-card-player-container'>
                        <ReactPlayer 
                            className='exercise-video-card-player'
                            url={`http://localhost:8000/video?video_name=${video_path}`} 
                            controls={false}   
                            loop={true} 
                            playing={true}
                            muted={true}
                            playsinline={true}
                            width='100%'
                            height='100%'
                        />
                    </div>
                    <div className='exercise-video-tips-container'>
                        <div className='exercise-video-tips-header'>
                            <span className='tips-icon'>📄</span>
                            <h4 className='exercise-video-tips-title'>Instrucciones de grabación</h4>
                        </div>
                        <div className='exercise-video-tips-grid'>
                            <div className='tip-item'>
                                <span className='tip-emoji'>📱</span>
                                <span>Graba en vertical</span>
                            </div>
                            <div className='tip-item'>
                                <span className='tip-emoji'>📏</span>
                                <span>Misma distancia de grabación que el video</span>
                            </div>
                            <div className='tip-item'>
                                <span className='tip-emoji'>👕</span>
                                <span>Usa ropa de color contraste con el fondo</span>
                            </div>
                            <div className='tip-item'>
                                <span className='tip-emoji'>💡</span>
                                <span>Buena iluminación</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};


export default ExerciseVideoCard;