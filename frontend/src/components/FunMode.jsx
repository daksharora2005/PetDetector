import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { FaMagic, FaRandom } from 'react-icons/fa';
import { predictFun } from '../api';

const FunMode = () => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [predictions, setPredictions] = useState(null);
    const [loading, setLoading] = useState(false);

    const onDrop = async (acceptedFiles) => {
        const f = acceptedFiles[0];
        setFile(f);
        setPreview(URL.createObjectURL(f));

        // Auto predict for fun text
        setLoading(true);
        try {
            const res = await predictFun(f);
            setPredictions(res.data.predictions);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    const { getRootProps, getInputProps } = useDropzone({
        onDrop,
        accept: { 'image/*': [] },
        maxFiles: 1
    });

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-4xl mx-auto pt-24 px-6 grid md:grid-cols-2 gap-8"
        >
            <div className="glass-panel p-6 h-fit">
                <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                    <FaMagic className="text-purple-400" />
                    Fun Identifier
                </h2>
                <div
                    {...getRootProps()}
                    className={`aspect-square rounded-xl border-2 border-dashed border-gray-600 hover:border-purple-500 overflow-hidden flex items-center justify-center cursor-pointer relative group
                        ${preview ? 'border-none' : ''}
                    `}
                >
                    <input {...getInputProps()} />
                    {preview ? (
                        <>
                            <img src={preview} alt="Fun" className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-black/60 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                <p className="text-white font-medium">Click to Change</p>
                            </div>
                        </>
                    ) : (
                        <div className="text-center p-4">
                            <FaRandom className="mx-auto text-4xl text-gray-500 mb-2" />
                            <p className="text-gray-400">Identify Anything!</p>
                        </div>
                    )}
                </div>
            </div>

            <div className="glass-panel p-6">
                <h3 className="text-xl font-bold mb-4 text-gray-200">AI Thinks it is...</h3>

                {loading ? (
                    <div className="space-y-4 animate-pulse">
                        {[1, 2, 3].map(i => (
                            <div key={i} className="h-10 bg-white/10 rounded-lg" />
                        ))}
                    </div>
                ) : predictions ? (
                    <div className="space-y-4">
                        {predictions.map((pred, i) => (
                            <motion.div
                                key={pred.label}
                                initial={{ x: 20, opacity: 0 }}
                                animate={{ x: 0, opacity: 1 }}
                                transition={{ delay: i * 0.1 }}
                                className="relative"
                            >
                                <div className="flex justify-between text-sm mb-1 z-10 relative">
                                    <span className="capitalize font-medium text-white">{pred.label}</span>
                                    <span className="text-purple-300">{(pred.probability * 100).toFixed(1)}%</span>
                                </div>
                                <div className="w-full h-3 bg-gray-800 rounded-full overflow-hidden">
                                    <motion.div
                                        initial={{ width: 0 }}
                                        animate={{ width: `${pred.probability * 100}%` }}
                                        className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                                    />
                                </div>
                            </motion.div>
                        ))}
                    </div>
                ) : (
                    <div className="flex items-center justify-center h-48 text-gray-500 italic">
                        Upload an image to see magic.
                    </div>
                )}
            </div>
        </motion.div>
    );
};

export default FunMode;
