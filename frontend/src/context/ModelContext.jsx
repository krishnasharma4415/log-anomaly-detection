import { createContext, useContext, useState } from 'react';

const ModelContext = createContext();

export function ModelProvider({ children }) {
    const [activeModel, setActiveModel] = useState({
        type: 'ml',
        name: 'XGBoost + SMOTE',
    });

    const updateActiveModel = (type, name) => {
        setActiveModel({ type, name });
    };

    return (
        <ModelContext.Provider value={{ activeModel, updateActiveModel }}>
            {children}
        </ModelContext.Provider>
    );
}

export function useModel() {
    const context = useContext(ModelContext);
    if (!context) {
        throw new Error('useModel must be used within ModelProvider');
    }
    return context;
}
