// src/components/ExerciseCard.js
import React from 'react';

const ExerciseCard = ({ name, imagePath, description }) => {

    function handleClick() {
        console.log(`Clicked on ${name}`);
        window.location.href = `/exercises?exercise=${formatName(name)}`;
    }

    function formatName(name) {
        return name.toLowerCase().replace(/ /g, '_');
    }

    return (
        <div className="exercise-card">
            <div className="exercise-card-grid">
                <div className="exercise-card-header">
                    <div className="exercise-card-content">
                        <h2 className="exercise-card-title">{name}</h2>
                        <p className="exercise-card-description">
                            {description}
                        </p>
                    </div>
                    <button className="exercise-card-button" onClick={handleClick}>
                        Probar ahora
                    </button>
                </div>
                <div className="exercise-card-image-container">
                    <div className="exercise-card-gradient" />
                    <img
                        src={`http://localhost:8000/image?image_name=${imagePath}`}
                        alt="Abstract design"
                        className="exercise-image"
                    />
                </div>
            </div>
        </div>
    );
};

export default ExerciseCard;

