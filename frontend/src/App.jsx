import React, { useState } from 'react';
import Navbar from './components/Navbar';
import LandingPage from './components/LandingPage';
import TrainingPanel from './components/TrainingPanel';
import InferencePanel from './components/InferencePanel';
import LiveWebcam from './components/LiveWebcam';
import FunMode from './components/FunMode';
import ChatWidget from './components/ChatWidget';

function App() {
  const [activeTab, setActiveTab] = useState('home');

  return (
    <div className="min-h-screen bg-slate-900 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-800 via-slate-900 to-black text-white selection:bg-brand-primary selection:text-white">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      <main className="container mx-auto">
        {activeTab === 'home' && <LandingPage onGetStarted={() => setActiveTab('train')} />}
        {activeTab === 'train' && <TrainingPanel setActiveTab={setActiveTab} />}
        {activeTab === 'predict' && <InferencePanel />}
        {activeTab === 'cam' && <LiveWebcam />}
        {activeTab === 'fun' && <FunMode />}
      </main>

      {/* Floating Chatbot */}
      <ChatWidget />
    </div>
  );
}

export default App;
