// src/components/ExerciseVideoCard.js
import React from 'react';
import ReactPlayer from 'react-player';

const ExerciseVideoCard = ({ name, description, video_path }) => {
    return (
        <div className='exercise-video-card'>
            <div className='exercise-video-card-grid'>
                <h1 className='section-title'>{name}</h1>
                <p className='exercise-video-card-description'>{description}</p>
                <div className='exercise-video-card-player-container'>
                    <ReactPlayer className='exercise-video-card-player'
                        url={`http://localhost:8000/video?video_name=${video_path}`} controls={true}
                        loop={true} playing={true} />
                </div>
            </div>
        </div>
    );
};

export default ExerciseVideoCard;

