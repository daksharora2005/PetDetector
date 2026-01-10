import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { FaCloudUploadAlt, FaCheck, FaSpinner, FaArrowRight, FaInfoCircle } from 'react-icons/fa';
import { uploadImages, startTraining, getStatus } from '../api';

const StepCard = ({ number, title, active }) => (
    <div className={`p-4 rounded-xl border border-white/10 flex items-center gap-3 transition-all ${active ? 'bg-brand-primary/20 border-brand-primary' : 'bg-white/5 opacity-50'}`}>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${active ? 'bg-brand-primary text-white' : 'bg-gray-700 text-gray-400'}`}>
            {number}
        </div>
        <span className="font-medium">{title}</span>
    </div>
);

const FileDrop = ({ label, onDrop, files }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': [] }
    });

    return (
        <div className="flex-1">
            <label className="block text-sm font-medium text-gray-400 mb-2">{label}</label>
            <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-6 flex flex-col items-center justify-center transition-all h-40 cursor-pointer relative group
          ${isDragActive ? 'border-brand-primary bg-brand-primary/10' : 'border-gray-600 hover:border-brand-primary/50 hover:bg-white/5'}
        `}
            >
                <input {...getInputProps()} />
                <FaCloudUploadAlt className="text-3xl text-gray-500 mb-2 group-hover:scale-110 transition-transform" />
                <p className="text-sm text-gray-400 text-center">
                    {files.length > 0
                        ? <span className="text-green-400 font-bold">{files.length} images selected</span>
                        : "Drop your pet images here"}
                </p>
                {files.length === 0 && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity">
                        <span className="text-white font-medium">Click to Browse</span>
                    </div>
                )}
            </div>
        </div>
    );
};

const TrainingPanel = ({ setActiveTab }) => {
    const [class1Name, setClass1Name] = useState('');
    const [class2Name, setClass2Name] = useState('');
    const [manualClass2, setManualClass2] = useState(false); // Track if user manually set class 2
    const [c1Files, setC1Files] = useState([]);
    const [c2Files, setC2Files] = useState([]);

    // Friendly UI State
    const [step, setStep] = useState(1); // 1: Setup, 2: Uploaded, 3: Training

    const [status, setStatus] = useState({ is_training: false, progress: 0, message: "Idle" });
    const [uploaded, setUploaded] = useState(false);

    // Smart Naming Logic
    useEffect(() => {
        if (!class1Name || manualClass2) return;

        const cleanName = class1Name.trim();
        let derivedName = cleanName;

        // "My Bo" -> "Bo"
        if (derivedName.toLowerCase().startsWith("my ")) {
            derivedName = derivedName.substring(3);
        }
        // "Bo the dog" -> "Bo" (Simple heuristic: take first word if > 2 words)
        else if (derivedName.split(' ').length > 2) {
            derivedName = derivedName.split(' ')[0];
        }

        // Capitalize 
        if (derivedName) {
            derivedName = derivedName.charAt(0).toUpperCase() + derivedName.slice(1);
            setClass2Name(`Not ${derivedName}`);
        } else {
            setClass2Name('');
        }
    }, [class1Name, manualClass2]);

    // Poll status & Redirect
    useEffect(() => {
        let interval;
        if (status.is_training) {
            interval = setInterval(async () => {
                try {
                    const res = await getStatus();
                    setStatus(res.data);

                    // Auto-Redirect Condition
                    if (!res.data.is_training && res.data.progress === 100) {
                        alert("Training Complete! Redirecting to Test Area...");
                        setActiveTab('predict'); // Switch tab
                    }
                } catch (e) { console.error(e); }
            }, 1000);
        }
        return () => clearInterval(interval);
    }, [status.is_training, setActiveTab]);

    const handleUpload = async () => {
        try {
            if (!class1Name) return alert("Please name your pet!");
            // Auto name class 2 if empty
            const finalClass2 = class2Name || `Not ${class1Name}`;

            if (c1Files.length < 5) return alert("Please upload at least 5 images of your pet.");

            // Check if user is PROVIDING background images or expecting defaults
            const usingDefaults = c2Files.length === 0;

            setStatus(s => ({ ...s, message: "Uploading your data..." }));

            // Simple 80/20 Split
            const splitFiles = (files) => {
                const splitIdx = Math.floor(files.length * 0.8);
                return [files.slice(0, splitIdx), files.slice(splitIdx)];
            };

            const [c1Train, c1Valid] = splitFiles(c1Files);
            await uploadImages(c1Train, 'train', class1Name);
            await uploadImages(c1Valid, 'valid', class1Name);

            if (!usingDefaults) {
                const [c2Train, c2Valid] = splitFiles(c2Files);
                await uploadImages(c2Train, 'train', finalClass2);
                await uploadImages(c2Valid, 'valid', finalClass2);
            } else {
                // The backend uses 'defaults' if folder empty, but we need to ensure the folder IS created with the right name
                // Currently backend auto-populates based on folder name passed in /train start
                // We rely on handleTrain passing the dynamic name.
            }

            setUploaded(true);
            setStep(2); // Move to next visual step
            setStatus(s => ({ ...s, message: usingDefaults ? "Pet images uploaded. Backgrounds will be auto-generated." : "Upload Complete. Ready to Train." }));

        } catch (e) {
            alert("Upload failed. Check console.");
            console.error(e);
        }
    };

    const handleTrain = async () => {
        try {
            setStep(3);
            await startTraining(class1Name, class2Name || `Not ${class1Name}`);
            setStatus(s => ({ ...s, is_training: true }));
        } catch (e) { console.error(e); }
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="max-w-5xl mx-auto pt-24 pb-10 px-6"
        >
            {/* Friendly Steps Header */}
            <div className="flex justify-between mb-8 gap-4">
                <StepCard number="1" title="Upload Pet Photos" active={step === 1} />
                <div className="h-1 flex-1 bg-white/10 self-center rounded-full" />
                <StepCard number="2" title="Prepare Data" active={step === 2} />
                <div className="h-1 flex-1 bg-white/10 self-center rounded-full" />
                <StepCard number="3" title="Train AI" active={step === 3} />
            </div>

            <div className="glass-panel p-8 mb-8 relative overflow-hidden">
                {/* Decoration */}
                <div className="absolute top-0 right-0 w-64 h-64 bg-brand-primary/10 rounded-full blur-3xl -translate-y-1/2 translate-x-1/2 pointer-events-none" />

                <h2 className="text-2xl font-bold mb-2 flex items-center gap-2">
                    <span className="bg-brand-primary/20 p-2 rounded-lg text-brand-primary text-xl">🚀</span>
                    Train Your Custom Model
                </h2>
                <p className="text-gray-400 mb-6 flex items-center gap-2">
                    <FaInfoCircle />
                    Teach the AI to recognize your specific pet. We'll handle the rest!
                </p>

                <div className="grid md:grid-cols-2 gap-8 mb-8">
                    {/* Class 1 Config */}
                    <div className="space-y-4">
                        <div>
                            <label className="text-sm font-semibold text-gray-300">Target Pet Name</label>
                            <input
                                type="text"
                                placeholder="e.g. 'My Bo'"
                                className="glass-input w-full mt-1"
                                value={class1Name}
                                onChange={e => {
                                    setClass1Name(e.target.value);
                                }}
                            />
                        </div>
                        <FileDrop
                            label={`Upload photos of ${class1Name || 'your pet'}`}
                            files={c1Files}
                            onDrop={setC1Files}
                        />
                    </div>

                    {/* Class 2 Config */}
                    <div className="space-y-4 opacity-80 hover:opacity-100 transition-opacity">
                        <div>
                            <label className="text-sm font-semibold text-gray-300 flex justify-between">
                                Negative Class (Optional)
                                <span className="text-xs text-brand-primary">Auto-Generated</span>
                            </label>
                            <input
                                type="text"
                                placeholder="e.g. 'Not Bo'"
                                className="glass-input w-full mt-1"
                                value={class2Name}
                                onChange={e => {
                                    setClass2Name(e.target.value);
                                    setManualClass2(true);
                                }}
                            />
                        </div>
                        <FileDrop
                            label={`Upload background photos (Optional)`}
                            files={c2Files}
                            onDrop={setC2Files}
                        />
                    </div>
                </div>

                <div className="flex justify-between items-center border-t border-white/10 pt-6">
                    <p className="text-sm text-gray-500 italic">
                        * Recommendation: Use at least 20 diverse images for best results.
                    </p>

                    <div className="flex gap-4">
                        {!uploaded && (
                            <button onClick={handleUpload} className="btn-secondary flex items-center gap-2">
                                <FaCloudUploadAlt /> Upload Data
                            </button>
                        )}

                        <button
                            onClick={handleTrain}
                            disabled={!uploaded || status.is_training}
                            className={`btn-primary flex items-center gap-2 ${(!uploaded || status.is_training) ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            {status.is_training ? <FaSpinner className="animate-spin" /> : <FaArrowRight />}
                            {status.is_training ? "Training in progress..." : "Start Training"}
                        </button>
                    </div>
                </div>
            </div>

            {/* Progress Section */}
            <AnimatePresence>
                {(status.is_training || status.progress > 0) && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="glass-panel p-6"
                    >
                        <div className="flex justify-between text-sm mb-2">
                            <span className="text-gray-400 flex items-center gap-2">
                                <FaSpinner className="animate-spin text-brand-secondary" />
                                {status.message}
                            </span>
                            <span className="font-bold text-brand-primary">{status.progress}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2.5 overflow-hidden">
                            <motion.div
                                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2.5 rounded-full"
                                style={{ width: `${status.progress}%` }}
                                layoutId="progress"
                            />
                        </div>
                        <p className="text-xs text-gray-500 mt-2 text-center">
                            The model is analyzed by the backend using PyTorch. Please wait while we optimize the weights.
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default TrainingPanel;
