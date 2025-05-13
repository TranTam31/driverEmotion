import React, { useState, useEffect } from 'react';

const DriversList = ({ drivers = [], selectedDriver, onSelectDriver, flaggedDriverIds = [] }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredDrivers, setFilteredDrivers] = useState([]);
  const [displayLimit, setDisplayLimit] = useState(3);

  // Xử lý lọc và sắp xếp tài xế mỗi khi dữ liệu thay đổi
  useEffect(() => {
    const filtered = drivers.filter(driver =>
      driver.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // Sắp xếp: flagged lên trước
    const sorted = [...filtered].sort((a, b) => {
      const aFlagged = flaggedDriverIds.includes(a.id);
      const bFlagged = flaggedDriverIds.includes(b.id);
      return (aFlagged === bFlagged) ? 0 : aFlagged ? -1 : 1;
    });

    setFilteredDrivers(sorted);
  }, [searchTerm, drivers, flaggedDriverIds]);

  // Tăng số lượng tài xế hiển thị khi nhấn nút "Xem thêm"
  const handleLoadMore = () => {
    setDisplayLimit(prevLimit => prevLimit + 3);
  };

  return (
    <div className="drivers-panel" style={{ width: '30%', padding: '5px 20px', borderRight: '1px solid #ddd' }}>
      <h2>Drivers</h2>
      
      <div className="search-container" style={{ marginBottom: '15px' }}>
        <input
          type="text"
          placeholder="Search drivers..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          style={{
            width: '100%',
            padding: '8px 12px',
            borderRadius: '4px',
            border: '1px solid #ddd',
            fontSize: '14px'
          }}
        />
      </div>

      <div 
        className="divider" 
        style={{ 
          borderTop: '1px solid #ddd', 
          margin: '5px 0', 
          width: '100%' 
        }}
      ></div>
      
      {/* Danh sách tài xế có giới hạn hiển thị và cuộn */}
      <div className="drivers-list" style={{ maxHeight: '500px', overflow: 'auto' }}>
        {filteredDrivers.slice(0, displayLimit).map(driver => {
          const isFlagged = flaggedDriverIds.includes(driver.id);

          return (
            <div
              key={driver.id}
              className={`driver-item ${selectedDriver && selectedDriver.id === driver.id ? 'selected' : ''}`}
              onClick={() => onSelectDriver(driver)}
              style={{
                padding: '10px',
                margin: '5px 0',
                border: '1px solid #ddd',
                borderRadius: '4px',
                cursor: 'pointer',
                backgroundColor: selectedDriver && selectedDriver.id === driver.id ? '#f0f8ff' : 'white',
                position: 'relative'
              }}
            >
              <div className="driver-name" style={{ fontWeight: 'bold', display: 'flex', alignItems: 'center' }}>
                {driver.name}
                {isFlagged && (
                  <span style={{
                    backgroundColor: '#ff4d4f',
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '8px',
                    fontSize: '12px',
                    marginLeft: '8px'
                  }}>
                    ⚠️ Alert
                  </span>
                )}
              </div>
              <div className="driver-info" style={{ fontSize: '14px', color: '#666' }}>
                Phone: {driver.phone} | License: {driver.license_number}
              </div>
            </div>
          );
        })}
        
        {filteredDrivers.length > displayLimit && (
          <button
            onClick={handleLoadMore}
            style={{
              width: '100%',
              padding: '10px',
              margin: '12px 0',
              backgroundColor: '#4a90e2', // Màu nền xanh dương đẹp mắt
              color: 'white', // Màu chữ trắng để tạo tương phản
              border: 'none',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px',
              fontWeight: '500', // Chữ hơi đậm một chút
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)', // Thêm đổ bóng nhẹ
              transition: 'all 0.3s ease', // Transition mượt mà cho hiệu ứng hover
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '5px' // Khoảng cách giữa icon và text nếu bạn thêm icon
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.backgroundColor = '#3a7dce'; // Màu xanh đậm hơn khi hover
              e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)'; // Đổ bóng lớn hơn khi hover
              e.currentTarget.style.transform = 'translateY(-2px)'; // Hiệu ứng nâng nhẹ khi hover
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.backgroundColor = '#4a90e2'; // Trở về màu ban đầu
              e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)'; // Trở về đổ bóng ban đầu
              e.currentTarget.style.transform = 'translateY(0)'; // Trở về vị trí ban đầu
            }}
          >
            <span>More</span> <span style={{ 
              backgroundColor: 'rgba(255,255,255,0.2)', 
              padding: '2px 8px', 
              borderRadius: '10px', 
              fontSize: '12px' 
            }}>
              {filteredDrivers.length - displayLimit} driver{(filteredDrivers.length - displayLimit) > 1 ? 's' : ''} left
            </span>
          </button>
        )}
        
        {filteredDrivers.length === 0 && (
          <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
            No drivers match.
          </div>
        )}
      </div>
    </div>
  );
};

export default DriversList;