import React from "react";

const InfoCard = ({ text }) => {

    return (
        <div className='info-card'>
            <div className='info-card-grid'>
                <p className='info-card-text'>{text}</p>
            </div>
        </div>
    );
};

export default InfoCard;
