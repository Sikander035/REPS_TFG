import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';
import ExerciseVideoCard from '../components/ExerciseVideoCard';
import UploadCard from '../components/UploadCard';

const Exercises = () => {

    const [exerciseName, setExerciseName] = useState('');
    const [exerciseDescription, setExerciseDescription] = useState('');
    const [exerciseVideo, setExerciseVideo] = useState('');

    useEffect(() => {
        const urlParams = new URLSearchParams(window.location.search);
        const exerciseParam = urlParams.get('exercise');
        fetch_exercise(formatName(exerciseParam)).then(data => {
            setExerciseName(data.name);
            setExerciseDescription(data.long_description);
            setExerciseVideo(data.original_video_path);
            console.log(data);
        });
    }, []);

    function formatName(name) {
        return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    async function fetch_exercise(exercise) {
        try {
            exercise = encodeURIComponent(exercise);
            let url = `http://localhost:8000/exercises?exercise_name=${exercise}`;
            const response = await fetch(url);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error fetching exercise:', error);
            return {};
        }
    }

    return (
        <div className='page-container'>
            <Navbar />
            <div className='exercise-container'>
                <ExerciseVideoCard name={exerciseName}
                    description={
                        <span dangerouslySetInnerHTML={{ __html: exerciseDescription }} />
                    }
                    video_path={exerciseVideo}
                />
                <UploadCard />
            </div>
            <Footer />
        </div>
    );
};

export default Exercises;
