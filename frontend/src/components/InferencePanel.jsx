import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { FaSearch, FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import { predictImage } from '../api';

const InferencePanel = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const onDrop = (acceptedFiles) => {
        const f = acceptedFiles[0];
        setFile(f);
        setPreview(URL.createObjectURL(f));
        setResult(null);
    };

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': [] },
        maxFiles: 1
    });

    const handlePredict = async () => {
        if (!file) return;
        setLoading(true);
        try {
            const res = await predictImage(file);
            setResult(res.data);
        } catch (e) {
            console.error(e);
            setResult({ error: "Failed to predict. Is the model trained?" });
        } finally {
            setLoading(false);
        }
    };

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-2xl mx-auto pt-24 px-6"
        >
            <div className="glass-panel p-8 text-center">
                <h2 className="text-3xl font-bold mb-8 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-pink-400">
                    Pet Detector
                </h2>

                {!preview ? (
                    <div
                        {...getRootProps()}
                        className={`border-2 border-dashed rounded-2xl h-64 flex flex-col items-center justify-center cursor-pointer transition-all
              ${isDragActive ? 'border-brand-primary bg-brand-primary/10' : 'border-gray-600 hover:border-gray-400'}
            `}
                    >
                        <input {...getInputProps()} />
                        <FaSearch className="text-4xl text-gray-500 mb-4" />
                        <p className="text-gray-400">Drag & Drop or Click to Test an Image</p>
                    </div>
                ) : (
                    <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-white/10">
                        <img src={preview} alt="Test" className="w-full h-64 object-cover" />
                        <button
                            onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                            className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 text-white rounded-full p-2 text-xs backdrop-blur-md"
                        >
                            Change
                        </button>
                    </div>
                )}

                <div className="mt-8">
                    <button
                        onClick={handlePredict}
                        disabled={!file || loading}
                        className={`btn-primary w-full text-lg ${(!file || loading) ? 'opacity-50' : ''}`}
                    >
                        {loading ? "Analyzing..." : "Identify Pet"}
                    </button>
                </div>

                <AnimatePresence>
                    {result && (
                        <motion.div
                            initial={{ opacity: 0, scale: 0.9 }}
                            animate={{ opacity: 1, scale: 1 }}
                            className="mt-8 p-6 bg-white/5 rounded-xl border border-white/10"
                        >
                            {result.error ? (
                                <p className="text-red-400">{result.error}</p>
                            ) : (
                                <div className="text-center">
                                    <p className="text-gray-400 mb-1">Result</p>
                                    <h3 className="text-4xl font-bold text-white mb-2">{result.class_name}</h3>
                                    <div className="inline-block px-3 py-1 bg-brand-primary/20 text-brand-primary rounded-full text-sm">
                                        {(result.confidence * 100).toFixed(1)}% Confidence
                                    </div>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>

            </div>
        </motion.div>
    );
};

export default InferencePanel;
