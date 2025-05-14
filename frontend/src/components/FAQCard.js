import React, { useState, useRef } from 'react';
import arrowDown from '../assets/arrow-down.png';

const FAQCard = ({ question, answer }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [contentHeight, setContentHeight] = useState(0);
    const answerRef = useRef(null);

    const toggleExpand = () => {
        if (isExpanded) {
            setContentHeight(0);
        } else {
            const scrollHeight = answerRef.current.scrollHeight;
            setContentHeight(scrollHeight);
        }
        setIsExpanded(!isExpanded);
    };

    return (
        <div className="FAQ-card">
            <div className="FAQ-card-grid">
                <div className="FAQ-card-question">
                    <h3 className="FAQ-card-title">{question}</h3>
                </div>

                <div className="FAQ-card-toggle" onClick={toggleExpand}>
                    <span className={`arrow ${isExpanded ? 'expanded' : ''}`}>
                        <img src={arrowDown} alt="Arrow down" />
                    </span>
                </div>
            </div>

            <div
                className={`FAQ-card-answer-wrapper ${isExpanded ? 'expanded' : ''}`}
                style={{ height: `${contentHeight}px` }}
            >
                <div
                    className={`FAQ-card-answer ${isExpanded ? 'visible' : ''}`}
                    ref={answerRef}
                >
                    <p>{answer}</p>
                </div>
            </div>
        </div>
    );
};

export default FAQCard;
