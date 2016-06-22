#include "three_d_grid.h"

using namespace std;

namespace std {
template<>
bool std::less<Eigen::Vector3i>::operator ()(const Eigen::Vector3i& a,const Eigen::Vector3i& b) const {
    for(size_t i=0;i<3;++i) {
        if(a[i]<b[i]) return true;
        if(a[i]>b[i]) return false;
    }
    return false;
}
}

BaseGrid::BaseGrid(string filename, int prec) : _cloud(new PointCloud) {
    pcl::io::loadPCDFile(filename.c_str(),*_cloud);
    double res = computeCloudResolution();
    PointT min_pt,max_pt;
    pcl::getMinMax3D(*_cloud,min_pt,max_pt);

    cout << "\n";
    cout << BOLD(FBLU("Loading the Data-Set:\n"));
    cout << "\t>> Filename: " << filename << "\n";
    cout << "\t>> Points: " << _cloud->size() << "\n";
    cout << "\t>> Min: (" << min_pt.x << "," << min_pt.y << "," << min_pt.z << ")\n";
    cout << "\t>> Max: (" << max_pt.x << "," << max_pt.y << "," << max_pt.z << ")\n";
    cout << "\t>> Average distance: " << res << "m\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "\n";

    bool found = false;
    int i = 2;
    while(found == false)
    {
        i--;
        if(getdigit(res,i) != 0)
            found = true;
    }
    _resolution = (100/prec)*pow(10,i);
    _inverse_resolution = 1./_resolution;

    _origin.x() = min_pt.x - 5*_resolution;
    _origin.y() = min_pt.y - 5*_resolution;
    _origin.z() = min_pt.z - 5*_resolution;

    _size.x() = ((max_pt.x+5*_resolution)-_origin.x())*_inverse_resolution;
    _size.y() = ((max_pt.y+5*_resolution)-_origin.y())*_inverse_resolution;
    _size.z() = ((max_pt.z+5*_resolution)-_origin.z())*_inverse_resolution;

    _num_cells = _size.x()*_size.y()*_size.z();

    size_t lastindex = filename.find_last_of(".");
    std::string directory = filename.substr(0,lastindex);
    manageDirectories(directory);

    pcl::PCLPointCloud2 out;
    pcl::toPCLPointCloud2(*_cloud,out);
    pcl::io::saveVTKFile("data_set.vtk",out);
}

double BaseGrid::computeCloudResolution() {
    double res = 0.0;
    int n_points = _cloud->size ();

    std::vector<int> indices (2);
    std::vector<float> sqr_distances (2);
    pcl::KdTreeFLANN<PointT> tree;
    tree.setInputCloud (_cloud);

    for (size_t i = 0; i < n_points; ++i)
        if(tree.nearestKSearch (i, 2, indices, sqr_distances) > 0)
            res += sqrt (sqr_distances[1]);

    return (res/n_points);
}

int BaseGrid::getdigit(double number, int digit) {
    div_t divresult = div(number/pow(10, digit),10);
    return  divresult.rem;
}

void BaseGrid::manageDirectories(std::string directory)
{
    struct stat info;
    if( stat( directory.c_str(), &info ) != 0 )
    {
        mkdir(directory.c_str(),0700);
    }
    else if( info.st_mode & S_IFDIR )
    {
        if(isDirectoryEmpty(directory.c_str()) == 0)
        {
            DIR *theFolder = opendir(directory.c_str());
            struct dirent *next_file;
            char filepath[256];

            while ( (next_file = readdir(theFolder)) != NULL )
            {
                if (0==strcmp(next_file->d_name, ".") || 0==strcmp(next_file->d_name, "..")) { continue; }
                char cwd[1024];
                getcwd(cwd, sizeof(cwd));
                sprintf(filepath, "%s/%s", directory.c_str(), next_file->d_name);
                remove(filepath);
            }
            closedir(theFolder);
        }
    }
    else
        printf( "%s is no directory\n", directory.c_str() );

    chdir(directory.c_str());
}

int BaseGrid::isDirectoryEmpty(const char *dirname)
{
    int n = 0;
    struct dirent *d;
    DIR *dir = opendir(dirname);
    if (dir == NULL)
        return 1;
    while ((d = readdir(dir)) != NULL) {
        if(++n > 2)
            break;
    }
    closedir(dir);
    if (n <= 2)
        return 1;
    else
        return 0;
}

float BaseGrid::solve(vector<float> &a) {
    float temp = a[0] + _resolution;
    if (temp <= a[1])
    {
        return temp;
    }
    else
    {
        temp = (a[0] + a[1] + sqrt(2*pow(_resolution,2)-pow(a[0]-a[1],2)))/2;
        if (temp <= a[2])
        {
            return temp;
        }
        else
            return ( a[0]+a[1]+a[2] + sqrt(pow(a[0]+a[1]+a[2],2) - 3*(pow(a[0],2)+pow(a[1],2)+pow(a[2],2)-pow(_resolution,2))) )/3;
    }
}

void BaseGrid::toIdx(float x, float y, float z, Eigen::Vector3i &idx) {
    idx.x() = floor((x - _origin.x())*_inverse_resolution);
    idx.y() = floor((y - _origin.y())*_inverse_resolution);
    idx.z() = floor((z - _origin.z())*_inverse_resolution);
}

void BaseGrid::toCell(float x, float y, float z, Eigen::Vector3i &idx) {
    idx.x() = floor((x - _resolution*0.5 - _origin.x())*_inverse_resolution);
    idx.y() = floor((y - _resolution*0.5 - _origin.y())*_inverse_resolution);
    idx.z() = floor((z - _resolution*0.5 - _origin.z())*_inverse_resolution);
}

void BaseGrid::toWorld(int i, int j, int k, Eigen::Vector3f &point) {
    point.x() = _origin.x() + i*_resolution + _resolution*0.5;
    point.y() = _origin.y() + j*_resolution + _resolution*0.5;
    point.z() = _origin.z() + k*_resolution + _resolution*0.5;
}

int BaseGrid::toInt(Eigen::Vector3i idx) {
    return (idx.z() + idx.y()*_size.z() + idx.x()*_size.y()*_size.z());
}

void BaseGrid::toIJK(int in, Eigen::Vector3i &idx) {
    div_t divresult = div(in,_size.x());
    idx.x() = divresult.rem;
    divresult = div(divresult.quot,_size.y());
    idx.y() = divresult.rem;
    divresult = div(divresult.quot,_size.z());
    idx.z() = divresult.rem;
}

FullGrid::FullGrid(string filename, int prec): BaseGrid(filename,prec) {

    cout << BOLD(FBLU("Building the 3D Grid (Full):\n"));
    cout << "\t>> Delta: " << _resolution << "m\n";
    cout << "\t>> Grid dimensions: (" << _size.x() << "," << _size.y() << "," << _size.z() << ")\t total: " << _num_cells << "\n";
    cout << "\t>> Origin: (" << _origin.x() << "," << _origin.y() << "," << _origin.z() << ")\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "\n";

    _data = new Cell[_num_cells];
    _cells = new Cell**[_size.x()];
    for(size_t x=0; x < _size.x(); x++) {
        _cells[x] = new Cell*[_size.y()];
        for(size_t y=0; y < _size.y(); y++) {
            _cells[x][y] = _data + y*_size.z() + x*_size.y()*_size.z();
            for(size_t z=0; z < _size.z(); z++) {
                Eigen::Vector3i idx(x,y,z);
                _data[z + y*_size.z() + x*_size.y()*_size.z()]._idx = idx;
                _data[z + y*_size.z() + x*_size.y()*_size.z()].setCenter(_origin,_resolution);
                if(x>0 && x<(_size.x()-1) && y>0 && y<(_size.y()-1) && z>0 && z<(_size.z()-1))
                    _data[z + y*_size.z() + x*_size.y()*_size.z()]._d = std::numeric_limits<float>::max() - 1;
            }
        }
    }

    for(size_t ii=0; ii < _cloud->size(); ii++) {
        Eigen::Vector3i idx;
        toIdx(_cloud->at(ii).x,_cloud->at(ii).y,_cloud->at(ii).z,idx);
        if(hasCell(idx)) {
            Cell& cell = _cells[idx.x()][idx.y()][idx.z()];
            cell._points.push_back(ii);
            float dist = euclideanDistance(cell._center,_cloud->at(ii));
            if(dist < cell._d) {
                cell._d = dist;
                cell._closest_point = ii;
            }
        }
        else {
            cout << "Error\n";
            cout << "Point (" << _cloud->at(ii).x << "," << _cloud->at(ii).y << "," << _cloud->at(ii).z << ")\t";
            cout << "Cell (" << idx.x() << "," << idx.y() << "," << idx.z() << "\n";
        }
    }

}

FullGrid::~FullGrid() {
    for(size_t x=0; x < _size.x(); x++) {
        delete [] _cells[x];
    }
    delete [] _cells;
    delete [] _data;
}

bool FullGrid::hasCell(const Eigen::Vector3i &idx)  {
    (idx.x() >= 0 && idx.x() <= _size.x() && idx.y() >= 0 && idx.y() <= _size.y() && idx.z() >= 0 && idx.z() <= _size.z()) ?  true : false;
}

int FullGrid::findNeighbors(Cell **neighbors, Cell *c) {
    int x = c->_idx.x();
    int y = c->_idx.y();
    int z = c->_idx.z();
    int xmin = (x-1<0) ? 0 : x-1;
    int xmax = (x+1>_size.x()-1) ? _size.x()-1 : x+1;
    int ymin = (y-1<0) ? 0 : y-1;
    int ymax = (y+1>_size.y()-1) ? _size.y()-1 : y+1;
    int zmin = (z-1<0) ? 0 : z-1;
    int zmax = (z+1>_size.z()-1) ? _size.z()-1 : z+1;
    int k=0;
    for(size_t xx=xmin; xx <= xmax; xx++)
        for(size_t yy=ymin; yy <= ymax; yy++)
            for(size_t zz=zmin; zz <= zmax; zz++)
                if(xx != x || yy != y || zz != z) {
                    neighbors[k] = &_cells[xx][yy][zz];
                    k++;
                }
    return k;
}

void FullGrid::computeDistanceFunction() {
    std::cout << "\n";
    std::cout << BOLD(FBLU("Level-Set Evolution:\n"));
    std::clock_t t0 = clock();
    //#####
    //# 1 #
    //#####
    for(size_t k = 1; k < _size.z()-1; k++)
        for(size_t j = 1; j < _size.y()-1; j++)
            for(size_t i = 1; i < _size.x()-1; i++) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 2 #
    //#####
    for(size_t k = 1; k < _size.z()-1; k++)
        for(size_t j = 1; j < _size.y()-1; j++)
            for(size_t i = _size.x()-2; i >= 1; i--) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 3 #
    //#####
    for(size_t k = 1; k < _size.z()-1; k++)
        for(size_t j = _size.y()-2; j >= 1 ; j--)
            for(size_t i = _size.x()-2; i >= 1; i--) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 4 #
    //#####
    for(size_t k = 1; k < _size.z()-1; k++)
        for(size_t j = _size.y()-2; j >= 1; j--)
            for(size_t i = 1; i < _size.x()-1; i++) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 5 #
    //#####
    for(size_t k = _size.z()-2; k >= 1; k--)
        for(size_t j = 1; j < _size.y()-1; j++)
            for(size_t i = 1; i < _size.x()-1; i++) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 6 #
    //#####
    for(size_t k = _size.z()-2; k >= 1; k--)
        for(size_t j = 1; j < _size.y()-1; j++)
            for(size_t i = _size.x()-2; i >= 1; i--) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 7 #
    //#####
    for(size_t k = _size.z()-2; k >= 1; k--)
        for(size_t j = _size.y()-2; j >= 1 ; j--)
            for(size_t i = _size.x()-2; i >= 1; i--) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################

    //#####
    //# 8 #
    //#####
    for(size_t k = _size.z()-2; k >= 1; k--)
        for(size_t j = _size.y()-2; j >= 1; j--)
            for(size_t i = 1; i < _size.x()-1; i++) {
                vector<float> Dmin (3);
                Dmin[0] = std::min(_cells[i-1][j][k]._d,_cells[i+1][j][k]._d);
                Dmin[1] = std::min(_cells[i][j-1][k]._d,_cells[i][j+1][k]._d);
                Dmin[2] = std::min(_cells[i][j][k-1]._d,_cells[i][j][k+1]._d);
                std::sort(Dmin.begin(),Dmin.end());
                _cells[i][j][k]._d = std::min(solve(Dmin),_cells[i][j][k]._d);
            }
    //#########################################################################
    std::clock_t t1 = clock();
    double elapsed_time1 = double(t1 - t0)/CLOCKS_PER_SEC;
    std::cout << "\tTime to compute Distance Function: " << elapsed_time1 << "s\n";
}

void FullGrid::computeGradient() {
    std::clock_t t0 = clock();
    for(size_t k = 2; k < _size.z()-2; k++)
        for(size_t j = 2; j < _size.y()-2; j++)
            for(size_t i = 2; i < _size.x()-2; i++) {
                float dx = (_cells[i+1][j][k]._d - _cells[i-1][j][k]._d)*_inverse_resolution*0.5;
                float dy = (_cells[i][j+1][k]._d - _cells[i][j-1][k]._d)*_inverse_resolution*0.5;
                float dz = (_cells[i][j][k+1]._d - _cells[i][j][k-1]._d)*_inverse_resolution*0.5;
                _cells[i][j][k]._gradient = Eigen::Vector3f(dx,dy,dz);
            }
    std::clock_t t1 = clock();
    double elapsed_time1 = double(t1 - t0)/CLOCKS_PER_SEC;
    std::cout << "\tTime to compute Distance Function Gradient: " << elapsed_time1 << "s\n";
}

void FullGrid::computeInitialSurface() {

    std::clock_t t0 = clock();
    CellQueue q;

    for(size_t k = 1; k < _size.z()-1; k++)
        for(size_t j = 1; j < _size.y()-1; j++)
            for(size_t i = 1; i < _size.x()-1; i++) {
                _cells[i][j][k]._u = -1;
                if(i==1 || i==_size.x()-2 || j==1 || j==_size.y()-2 || k==1 || k==_size.z()-2) {
                    _cells[i][j][k]._u = 0;
                    q.push(&_cells[i][j][k]);
                }
            }

    Cell* neighbors[26];
    bool stop = false;
    while (stop == false) {
        Cell* current = q.top();
        q.pop();
        int k = findNeighbors(neighbors,current);
        bool boundary_point = false;
        vector<Cell*> new_front;
        for(size_t ii=0; ii < k; ii++) {
            Cell* neighbor = neighbors[ii];
            if(neighbor->_u == -1) {
                if(neighbor->_d <= current->_d)
                    new_front.push_back(neighbor);
                else
                    boundary_point = true;
            }
        }

        if(boundary_point == true) {
            current->_u = 0;
        }
        else {
            current->_u = 1;
            for(vector<Cell*>::iterator it = new_front.begin(); it != new_front.end(); it++) {
                Cell* temp = *it;
                q.push(temp);
                temp->_u = 0;
            }
        }

        if(current->_d < _resolution)
            stop = true;
    }

    while(!q.empty()) {
        q.top()->_u = 0;
        q.pop();
    }
    std::clock_t t1 = clock();
    double elapsed_time1 = double(t1 - t0)/CLOCKS_PER_SEC;
    std::cout << "\tTime to compute Initial Guess: " << elapsed_time1 << "s\n";
}

void FullGrid::evolve(size_t num_iter, float deltaT) {
    writeDataToFile();
    std::cout << "\t>> Iterations: " << num_iter << "\n";
    std::cout << "\t>> Time step: " << deltaT << "\n";
    std::cerr << "\t";
    std::clock_t t0 = clock();
    vector<float> U_new(_num_cells,0);
    for(size_t iter=1; iter <= num_iter; iter++) {

        for(size_t k = 2; k < _size.z()-2; k++)
            for(size_t j = 2; j < _size.y()-2; j++)
                for(size_t i = 2; i < _size.x()-2; i++) {
                    float sol=0;
                    float DU[3];
                    DU[0] = (_cells[i+1][j][k]._u-_cells[i-1][j][k]._u)*_inverse_resolution*0.5;
                    DU[1] = (_cells[i][j+1][k]._u-_cells[i][j-1][k]._u)*_inverse_resolution*0.5;
                    DU[2] = (_cells[i][j][k+1]._u-_cells[i][j][k-1]._u)*_inverse_resolution*0.5;
                    float modDU = sqrt(pow(DU[0],2)+pow(DU[1],2)+pow(DU[2],2));
                    float sigma[3][2];
                    if(sqrt(pow(DU[0],2)+pow(DU[2],2)) != 0) {
                        sigma[0][0] = (-DU[2])/sqrt(pow(DU[0],2)+pow(DU[2],2));
                        sigma[1][0] = 0;
                        sigma[2][0] = DU[0]/sqrt(pow(DU[0],2)+pow(DU[2],2));
                        sigma[0][1] = (-DU[0]*DU[1])/(sqrt(pow(DU[0],2)+pow(DU[2],2))*modDU);
                        sigma[1][1] = sqrt(pow(DU[0],2)+pow(DU[2],2))/modDU;
                        sigma[2][1] = (-DU[1]*DU[2])/(sqrt(pow(DU[0],2)+pow(DU[2],2))*modDU);
                    }
                    else {
                        sigma[0][0] = 1;
                        sigma[1][0] = 0;
                        sigma[2][0] = 0;
                        sigma[0][1] = 0;
                        sigma[1][1] = 0;
                        sigma[2][1] = 1;
                    }
                    float d[2][4]={1,-1,1,-1,
                                   1,1,-1,-1};
                    Eigen::Vector3f center;
                    toWorld(i,j,k,center);
                    for(size_t idx=0; idx < 4; idx++) {
                        float xp = center.x() + _cells[i][j][k]._gradient.x()*deltaT + std::sqrt(2*deltaT*_cells[i][j][k]._d)*(sigma[0][0]*d[0][idx]+sigma[0][1]*d[1][idx]);
                        float yp = center.y() + _cells[i][j][k]._gradient.y()*deltaT + std::sqrt(2*deltaT*_cells[i][j][k]._d)*(sigma[1][0]*d[0][idx]+sigma[1][1]*d[1][idx]);
                        float zp = center.z() + _cells[i][j][k]._gradient.z()*deltaT + std::sqrt(2*deltaT*_cells[i][j][k]._d)*(sigma[2][0]*d[0][idx]+sigma[2][1]*d[1][idx]);

                        sol += interp(xp,yp,zp);
                    }

                    U_new[i + _size.x()*j + _size.x()*_size.y()*k] = sol/4;
                }

        for(size_t k = 2; k < _size.z()-2; k++)
            for(size_t j = 2; j < _size.y()-2; j++)
                for(size_t i = 2; i < _size.x()-2; i++)
                    _cells[i][j][k]._u = U_new[i + _size.x()*j + _size.x()*_size.y()*k];
        writeDataToFile(iter);
    }
    std::cerr << "\n";
    std::clock_t t1 = clock();
    double elapsed_time1 = double(t1 - t0)/CLOCKS_PER_SEC;
    std::cout << "\tTime to compute Surface Evolution: " << elapsed_time1 << "s\n";
}

float FullGrid::interp(float xp, float yp, float zp) {
    Eigen::Vector3i idx;
    toCell(xp,yp,zp,idx);

    Eigen::Vector3f point;
    toWorld(idx.x(),idx.y(),idx.z(),point);

    float u00 = (xp - point.x())*_inverse_resolution*(_cells[idx.x()+1][idx.y()][idx.z()]._u - _cells[idx.x()][idx.y()][idx.z()]._u) + _cells[idx.x()][idx.y()][idx.z()]._u;
    float u01 = (xp - point.x())*_inverse_resolution*(_cells[idx.x()+1][idx.y()+1][idx.z()]._u - _cells[idx.x()][idx.y()+1][idx.z()]._u) + _cells[idx.x()][idx.y()+1][idx.z()]._u;
    float u0  = (yp - point.y())*_inverse_resolution*(u01 - u00) + u00;

    float u10 = (xp - point.x())*_inverse_resolution*(_cells[idx.x()+1][idx.y()][idx.z()+1]._u - _cells[idx.x()][idx.y()][idx.z()+1]._u) + _cells[idx.x()][idx.y()][idx.z()+1]._u;
    float u11 = (xp - point.x())*_inverse_resolution*(_cells[idx.x()+1][idx.y()+1][idx.z()+1]._u - _cells[idx.x()][idx.y()+1][idx.z()+1]._u) + _cells[idx.x()][idx.y()+1][idx.z()+1]._u;
    float u1  = (yp - point.y())*_inverse_resolution*(u11 - u10) + u10;

    return ((zp - point.z())*_inverse_resolution*(u1 - u0) + u0);
}

void FullGrid::writeDataToFile(size_t iter) {
    vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
    imageData->SetDimensions(_size.x(),_size.y(),_size.z());
    imageData->SetOrigin(_origin.x() + _resolution/2,_origin.y() + _resolution/2,_origin.z() + _resolution/2);
    imageData->SetSpacing(_resolution,_resolution,_resolution);
    imageData->SetExtent(0,_size.x()-1,0,_size.y()-1,0,_size.z()-1);
#if VTK_MAJOR_VERSION <= 5
    imageData->SetNumberOfScalarComponents(1);
    imageData->SetScalarTypeToFloat();
#else
    imageData->AllocateScalars(VTK_FLOAT, 1);
#endif

    vtkSmartPointer<vtkFloatArray> dist = vtkSmartPointer<vtkFloatArray>::New();
    dist->SetName("distance");

    vtkSmartPointer<vtkFloatArray> s_dist = vtkSmartPointer<vtkFloatArray>::New();
    s_dist->SetName("signed_dist");

    for(int z = 0; z < _size.z(); z++)
        for(int y = 0; y < _size.y(); y++)
            for(int x = 0; x < _size.x(); x++) {
                s_dist->InsertNextValue(_cells[x][y][z]._u);
                dist->InsertNextValue(_cells[x][y][z]._d);
            }

    imageData->GetPointData()->AddArray(dist);
    imageData->GetPointData()->AddArray(s_dist);
    imageData->Update();

    ostringstream file;
    file << "surface_reconstruction_" << iter << ".vti";
    vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
    writer->SetFileName(file.str().c_str());
#if VTK_MAJOR_VERSION <= 5
    writer->SetInputConnection(imageData->GetProducerPort());
#else
    writer->SetInputData(imageData);
#endif
    writer->Write();
}

SparseGrid::SparseGrid (string filename, int prec) : BaseGrid(filename,prec) {
    for(size_t ii=0; ii < _cloud->size(); ii++) {
        Eigen::Vector3i idx;
        toIdx(_cloud->at(ii).x,_cloud->at(ii).y,_cloud->at(ii).z,idx);
        if(hasCell(idx)) {
            Cell* cell = _cells[idx];
            cell->_points.push_back(ii);
            float dist = euclideanDistance(cell->_center,_cloud->at(ii));
            if(dist < cell->_d) {
                cell->_d = dist;
                cell->_closest_point = ii;
            }
        }
        else {
            Cell* cell = new Cell(idx);
            cell->setCenter(_origin,_resolution);
            cell->_points.push_back(ii);
            float dist = euclideanDistance(cell->_center,_cloud->at(ii));
            if(dist < cell->_d) {
                cell->_d = dist;
                cell->_closest_point = ii;
            }
            map<Eigen::Vector3i,Cell*>::iterator it = _cells.begin();
            _cells.insert(it,std::pair<Eigen::Vector3i,Cell*>(idx,cell));
        }

    }

    cout << BOLD(FBLU("Building the 3D Grid (Sparse):\n"));
    cout << "\t>> Delta: " << _resolution << "m\n";
    cout << "\t>> Grid dimensions: (" << _size.x() << "," << _size.y() << "," << _size.z() << ")\t total: " << _num_cells << "\n";
    cout << "\t>> Origin: (" << _origin.x() << "," << _origin.y() << "," << _origin.z() << ")\n";
    cout << "\t>> Active cells: " << _cells.size() << "\n";
    cout << "--------------------------------------------------------------------------------\n";
    cout << "\n";
}

bool SparseGrid::hasCell(const Eigen::Vector3i &idx) {
    map<Eigen::Vector3i,Cell*>::iterator it = _cells.find(idx);
    (it != _cells.end()) ? true : false;
}

int SparseGrid::findNeighbors(Cell **neighbors, Cell *c) {
    int x = c->_idx.x();
    int y = c->_idx.y();
    int z = c->_idx.z();
    int xmin = (x-1<0) ? 0 : x-1;
    int xmax = (x+1>_size.x()-1) ? _size.x()-1 : x+1;
    int ymin = (y-1<0) ? 0 : y-1;
    int ymax = (y+1>_size.y()-1) ? _size.y()-1 : y+1;
    int zmin = (z-1<0) ? 0 : z-1;
    int zmax = (z+1>_size.z()-1) ? _size.z()-1 : z+1;
    int k=0;
    for(size_t xx=xmin; xx <= xmax; xx++)
        for(size_t yy=ymin; yy <= ymax; yy++)
            for(size_t zz=zmin; zz <= zmax; zz++)
                if(xx != x || yy != y || zz != z) {
                    Eigen::Vector3i idx(xx,yy,zz);
                    if(hasCell(idx)) {
                        neighbors[k] = _cells[idx];
                        k++;
                    }
                    else {
                        Cell* cell = new Cell(idx);
                        cell->setCenter(_origin,_resolution);
                        map<Eigen::Vector3i,Cell*>::iterator it = _cells.begin();
                        _cells.insert(it,std::pair<Eigen::Vector3i,Cell*>(idx,cell));
                        neighbors[k] = _cells[idx];
                        k++;
                    }
                }
    return k;
}
