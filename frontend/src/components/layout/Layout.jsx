import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import TopBar from './TopBar';

export default function Layout() {
  return (
    <div className="min-h-screen bg-slate-900 text-neutral-primary">
      <Sidebar />
      <div className="ml-64">
        <TopBar />
        <main className="p-8 text-neutral-primary">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
