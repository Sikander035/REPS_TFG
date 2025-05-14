// src/components/UploadCard.js
import React from 'react';
import FileUploader from './FileUploader';

const UploadCard = () => {

    return (
        <div className='upload-card'>
            <h1 className='section-title'>¡Sube tu video!</h1>
            <div className='upload-card-grid'>
                <FileUploader />
            </div>
        </div>
    );
};

export default UploadCard;

