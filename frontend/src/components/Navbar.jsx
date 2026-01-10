import React from 'react';
import { FaPaw } from 'react-icons/fa';
import { motion } from 'framer-motion';

const Navbar = ({ activeTab, setActiveTab }) => {
    const tabs = [
        { id: 'home', label: 'Home' },
        { id: 'train', label: 'Train Model' },
        { id: 'predict', label: 'Detector' },
        { id: 'cam', label: 'Live Guard' },
        { id: 'fun', label: 'Fun Mode' },
    ];

    return (
        <div className="w-full fixed top-0 z-50 px-6 py-4">
            <div className="glass-panel max-w-7xl mx-auto px-6 py-3 flex justify-between items-center">
                <div className="flex items-center gap-3">
                    <motion.div
                        whileHover={{ rotate: 20 }}
                        className="text-brand-primary text-3xl"
                    >
                        <FaPaw />
                    </motion.div>
                    <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-violet-400">
                        PetDetector AI
                    </h1>
                </div>

                <div className="flex gap-2">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`px-4 py-2 rounded-lg transition-all text-sm font-medium ${activeTab === tab.id
                                ? 'bg-brand-primary/20 text-brand-primary border border-brand-primary/50'
                                : 'text-gray-400 hover:text-white hover:bg-white/5'
                                }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default Navbar;
